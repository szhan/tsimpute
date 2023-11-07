""" Functions for adding new sample paths to an existing tree sequence. """
from dataclasses import dataclass, field
import tqdm

import numpy as np


@dataclass(frozen=True)
class SamplePath:
    """
    Convenience class for storing the sample paths of an individual.
    Each sample path is defined by a list of ids of nodes in a tree sequence.

    Definition of a valid `SamplePath` object:
    1. The sizes of the `nodes` and `site_positions` attributes are equal.
    2. The `site_positions` attribute is sorted in ascending order.

    individual: Name of individual.
    nodes: Sample path (list of node ids).
    site_positions: Site positions corresponding to the path.
    is_valid: Boolean indicating whether the path is valid.
    metadata: Metadata associated with the path (optional).
    """
    individual: str
    nodes: np.ndarray
    site_positions: np.ndarray
    metadata: dict = None
    is_valid: bool = field(init=False)

    def __len__(self):
        return(self.nodes.size)

    def __post_init__(self):
        object.__setattr__(self, 'is_valid', False)
        is_nodes_site_positions_equal_length = self.nodes.size == self.site_positions.size
        is_site_positions_sorted = np.all(self.site_positions[:-1] < self.site_positions[1:])
        if is_nodes_site_positions_equal_length and is_site_positions_sorted:
            object.__setattr__(self, 'is_valid', True)


def get_switch_mask(path):
    """
    Called by `get_switch_site_positions` and `get_num_switches`.

    :param SamplePath path: Sample path.
    :return: Indicators of whether the sample path switches at each site.
    :rtype: numpy.ndarray(dtype=bool)
    """
    is_switch = np.zeros(len(path), dtype=bool)
    is_switch[1:] = np.invert(np.equal(path.nodes[1:], path.nodes[:-1]))
    return(is_switch)


def get_switch_site_positions(path):
    """
    Get the site positions where the individual switches.

    :param SamplePath path: Sample path.
    :return: Site positions where the sample path switches.
    :rtype: numpy.ndarray
    """
    is_switch = get_switch_mask(path)
    return(path.site_positions[is_switch])


def get_num_switches(path):
    """
    :param SamplePath path: Sample path.
    :return: Number of switches in the sample path.
    :rtype: int
    """
    return(np.sum(get_switch_mask(path)))


def add_individuals_to_tree_sequence(ts, paths, individual_names, metadata=None):
    """
    Add individuals (each of which have sample paths) to an existing ts.

    Assumptions:
    1. All the individuals are diploid.
    2. The number of paths is twice the number of individuals.

    :param tskit.TreeSequence ts: Tree sequence to which the individuals are added.
    :param numpy.ndarray paths: Matrix of paths (samples by sites).
    :param list individual_names: List of names of the individuals.
    :param list metadata: list of metadata dict for the individuals.
    :return: Tree sequence with the newly added individuals.
    :rtype: tskit.TreeSequence
    """
    if ts.num_sites != paths.shape[1]:
        raise ValueError("Lengths of ts and paths are not equal.")
    if not np.all(np.isin(paths, np.arange(ts.num_nodes))):
        raise ValueError("Not all node ids in the paths are in the ts.")
    if paths.shape[0] != 2 * len(individual_names):
        raise ValueError("Number of paths is not twice the number of individuals.")

    new_tables = ts.dump_tables()

    # Initialise arrays to store data for new edges.
    new_edges_left_coords = np.array([], dtype=np.float64)
    new_edges_right_coords = np.array([], dtype=np.float64)
    new_edges_parent_nodes = np.array([], dtype=np.int32)
    new_edges_child_nodes = np.array([], dtype=np.int32)

    num_individuals = paths.shape[0] // 2
    for i in tqdm.tqdm(np.arange(num_individuals)):
        metadata_path = f"\"name\": \"{individual_names[i]}\", "
        metadata_path += f"\"status\": \"imputed\", "
        metadata_path += f"\"recomb\": \"uniform\""

        path_1 = SamplePath(
            individual=individual_names[i],
            nodes=paths[2 * i, :],
            site_positions=ts.sites_position,
            metadata=metadata_path.encode('ascii')
        )
        path_2 = SamplePath(
            individual=individual_names[i],
            nodes=paths[2 * i + 1, :],
            site_positions=ts.sites_position,
            metadata=metadata_path.encode('ascii')
        )

        assert path_1.is_valid
        assert path_2.is_valid

        # Add an individual to the individuals table.
        metadata_ind = metadata[i] if metadata is not None else None
        new_ind_id = new_tables.individuals.add_row(metadata=metadata_ind)

        for p in [path_1, path_2]:
            # Add a new sample node to the nodes table.
            new_node_id = new_tables.nodes.add_row(
                flags=1, # Flag for a sample
                time=-1, # Arbitrarily set to be younger than samples at t = 0 in ts
                population=0,   # TODO: Associate it with a specific population
                individual=new_ind_id,
                metadata=p.metadata,
            )
            new_node_id = np.int32(new_node_id)

            # Keep new edges to the expanded edges table.
            is_switch = get_switch_mask(p)
            parent_at_switch = p.nodes[is_switch]
            pos_at_switch = p.site_positions[is_switch]

            if len(pos_at_switch) > 0:
                if pos_at_switch[0] == 0:
                    raise ValueError("Switch cannot occur at the first site in the sequence.")
                if pos_at_switch[-1] == ts.sequence_length - 1:
                    raise ValueError("Switch cannot occur at the last site in the sequence.")

            # Recall that edge span is expressed as half-open,
            # so the right position is exclusive.
            pos = np.concatenate(([0], pos_at_switch, [ts.sequence_length]))
            parent_nodes = np.concatenate(([p.nodes[0]], parent_at_switch))
            left_coords = pos[:-1]
            right_coords = pos[1:]
            child_nodes = np.repeat(new_node_id, len(parent_nodes))

            new_edges_left_coords = np.concatenate((new_edges_left_coords, left_coords))
            new_edges_right_coords = np.concatenate((new_edges_right_coords, right_coords))
            new_edges_parent_nodes = np.concatenate((new_edges_parent_nodes, parent_nodes))
            new_edges_child_nodes = np.concatenate((new_edges_child_nodes, child_nodes))

    # Add the new edges all at once.
    assert len(new_edges_left_coords) == len(new_edges_right_coords)
    assert len(new_edges_left_coords) == len(new_edges_parent_nodes)
    assert len(new_edges_left_coords) == len(new_edges_child_nodes)

    new_tables.edges.append_columns(
        left=new_edges_left_coords,
        right=new_edges_right_coords,
        parent=new_edges_parent_nodes,
        child=new_edges_child_nodes,
    )

    new_tables.sort()
    new_ts = new_tables.tree_sequence()

    return new_ts
