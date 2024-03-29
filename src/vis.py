import math
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Functions for making static visualizations
def plot_sample_path(path, site_pos, tracks=None, window=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.plot(
        site_pos,
        path,
    )
    # Add tracks
    if tracks is not None:
        for i in np.arange(len(tracks)):
            ax.plot(
                tracks[i][0],
                np.repeat(-(i + 1) * 1_000, len(tracks[i][0])),
                marker="|",
                color=tracks[i][1],
                linestyle=""
            )
    if window is not None:
        assert len(window) == 2
        ax.set_xlim(window[0], window[1])
    ax.set_ylabel("Index of sample")
    ax.set_xlabel("Genomic position");


def compare_sample_paths(path_1, path_2):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5,))

    bool_identical_samples = np.equal(path_1, path_2)
    bool_different_samples = np.invert(bool_identical_samples)

    sample_indices = np.arange(len(path_1))

    ax.plot(
        sample_indices[bool_identical_samples],
        path_1[bool_identical_samples],
        color="black", marker="o", linestyle=""
    )
    ax.plot(
        sample_indices[bool_different_samples],
        path_1[bool_different_samples],
        color="blue", marker="o", linestyle=""
    )
    ax.plot(
        sample_indices[bool_different_samples],
        path_2[bool_different_samples],
        color="orange", marker="o", linestyle=""
    )

    ax.set_ylabel("Index of sample")
    ax.set_xlabel("Index of site");


def plot_iqs_versus_maf(
    df,
    method,
    min_iqs=0.90,
    max_maf=0.50,
    subsample_frac=0.01,
    out_png_file=None,
    dpi=100,
):
    assert 0.0 <= subsample_frac <= 1.0

    subsample_size = math.ceil(df.shape[0] * subsample_frac)
    subsample = np.random.choice(np.arange(df.shape[0]), subsample_size)

    values = np.vstack([df["ref_ma_freq"][subsample], df["iqs"][subsample]])
    kernel = stats.gaussian_kde(values)
    x = kernel(np.vstack([df["ref_ma_freq"], df["iqs"]]))

    num_sites_min_iqs = np.sum(df["iqs"] >= min_iqs)
    prop_sites_min_iqs = num_sites_min_iqs / float(df.shape[0])

    fig, ax = plt.subplots(figsize=(7, 7,))

    ax.set_title(
        f"{method}"
        "\n"
        f"% sites with min IQS: {round(prop_sites_min_iqs * 100.0, 2)}",
        size=20
    )
    ax.set_xlim([0, max_maf])
    ax.set_ylabel("IQS", size=20)
    ax.set_xlabel("MAF", size=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    g = sns.scatterplot(
        y="iqs",
        x="ref_ma_freq",
        data=df,
        c=x,
        cmap="viridis",
        #x_jitter=True,
        ax=ax,
        alpha=0.2
    )

    if out_png_file is not None:
        g.get_figure().savefig(out_png_file, dpi=dpi)


# Functions for making interactive visualizations
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, NumeralTickFormatter, Legend
from bokeh.palettes import brewer
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application


def _get_data():
    raise NotImplementedError


def create_sample_path_vis_app(
    path,
    ts,
    tracks,
    legend_labels,
    controls,
    matrix=None,
    range_node_id=None,
    range_site_id=None, # TODO: Implement.
    range_site_pos=None,
):
    """
    TODO: Add docstring.
    TODO: Major refactor.

    :param util.SamplePath path: Sample path.
    :param tskit.TreeSequence ts: Tree sequence containing parent nodes of the sample path.
    :param list(dict) tracks: List of dicts with keys 'site_pos' and 'site_base'.
    :param tuple(int, int) range_nodes: Range of parent node ids to display.
    :param tuple(float, float) range_site_pos: Range of site positions to display.
    :return: Bokeh application.
    :rtype: bokeh.application.Application
    """
    def modify_doc(doc):
        #ctrl_args = {name: ctrl.value for name, ctrl in controls.items()}
        is_sample = np.array(ts.nodes_flags[path.nodes], dtype=bool)
        path_site_ids = np.arange(len(path))
        source_all = ColumnDataSource(data=dict(
            node_id=path.nodes,
            site_id=path_site_ids,
            site_pos=path.site_positions,
        ))
        source_sample = ColumnDataSource(data=dict(
            node_id=path.nodes[is_sample],
            site_id=path_site_ids[is_sample],
            site_pos=path.site_positions[is_sample],
        ))
        source_nonsample = ColumnDataSource(data=dict(
            node_id=path.nodes[~is_sample],
            site_id=path_site_ids[~is_sample],
            site_pos=path.site_positions[~is_sample],
        ))
        if matrix is not None:
            source_matrix = ColumnDataSource(data=dict(
                node_id=matrix['node_id'].values,
                site_id=matrix['site_id'].values,
                site_pos=matrix['site_pos'].values,
                prob=matrix['prob'].values,
            ))

        # TODO: Add more info about the parent nodes.
        TOOLTIPS = [
            ("Parent node id", "@node_id"),
            ("Site id", "@site_id"),
            ("Site position", "@site_pos"),
        ]

        # Create the main plot
        if matrix is not None:
            x_axis_label = 'Site index'
            offset = 1
            x_range = Range1d(
                0 - offset, len(path) + offset,
                bounds=(0 - offset, len(path) + offset),
            )
        else:
            x_axis_label = 'Genomic position'
            offset = 10**3
            min_site_pos = 1
            max_site_pos = ts.sequence_length
            if range_site_pos is not None:
                min_site_pos = range_site_pos[0]
                max_site_pos = range_site_pos[1]
            x_range = Range1d(
                min_site_pos - offset, max_site_pos + offset,
                bounds=(min_site_pos - offset, max_site_pos + offset),
            )

        p1 = figure(
            height=400, width=800,
            x_axis_label=x_axis_label,
            y_axis_label='Parent node id',
            tooltips=TOOLTIPS,
        )
        p1.x_range = x_range
        min_node_id = 0
        max_node_id = ts.num_nodes - 1
        if range_node_id is not None:
            min_node_id = range_node_id[0]
            max_node_id = range_node_id[1]
        p1.y_range = Range1d(
            min_node_id, max_node_id,
            bounds=(min_node_id, max_node_id),
        )
        p1.xaxis.axis_label_text_font_style = 'normal'
        p1.yaxis.axis_label_text_font_style = 'normal'
        p1.xaxis.axis_label_text_font_size = '14pt'
        p1.yaxis.axis_label_text_font_size = '14pt'
        p1.xaxis.formatter = NumeralTickFormatter(format='0.00a')
        p1.grid.visible = False

        # Show probability matrix
        if matrix is not None:
            p1.rect(
                x='site_id',
                y='node_id',
                source=source_matrix,
                width=1, height=5,
                fill_color=linear_cmap(
                    'prob',
                    palette=brewer['Purples'][9],
                    low=0,
                    high=1,
                ),
                line_color=None,
                fill_alpha=0.25,
            )

        # Show copying path
        x_axis_choice = 'site_id' if matrix is not None else 'site_pos'
        r1 = p1.step(
            x=x_axis_choice, y='node_id', source=source_all,
            line_width=2, line_color='purple', mode='after',
        )
        r2 = p1.square(
            x=x_axis_choice, y='node_id', source=source_sample,
            fill_color='black', size=6, line_width=0,
        )
        r3 = p1.circle(
            x=x_axis_choice, y='node_id', source=source_nonsample,
            fill_color='gray', size=6, line_width=0,
        )

        # Add legend
        legend1 = Legend(items=[
            ('Copying path', [r1]),
            ('Sample nodes', [r2]),
            ('Non-sample nodes', [r3]),
        ], location='center', border_line_alpha=0,)
        p1.add_layout(legend1, 'right')

        # Show additional data
        p2 = figure(
            height=100, width=800,
            x_axis_label='', y_axis_label='',
            x_range=p1.x_range, y_range=p1.y_range,
        )

        _base_color_map = {
            -2: '#7f7f7f', -1: '#ffffff',
            0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'
        }
        i = 0
        for track in tracks:
            # TODO: Use another renderer. I suspect this is too costly.
            p2.vbar(
                x=track['site_pos'],
                top=np.repeat(i + 0.8, len(track['site_pos'])),
                bottom=np.repeat(i, len(track['site_pos'])),
                color=[_base_color_map[base] for base in track['site_base']],
                width=1,
            )
            i += 1

        p2.y_range = Range1d(0, len(tracks), bounds=(0, len(tracks)))
        p2.xaxis.visible = False
        p2.yaxis.major_tick_line_color = None
        p2.yaxis.minor_tick_line_color = None
        p2.yaxis.major_label_text_color = None
        p2.grid.visible = False

        # Add legend
        renderers = [(l, [r]) for l, r in zip(legend_labels, p2.renderers)]
        renderers.reverse()
        legend2 = Legend(
            items=renderers,
            location='center', border_line_alpha=0,
            glyph_height=0, glyph_width=0,
        )
        legend2.click_policy = 'mute'
        p2.add_layout(legend2, 'right')

        # Define on-change behavior
        # TODO: Use toggle instead.
        for ctrl_name, ctrl in controls.items():
            def update(attr, old, new):
                control_args = {name: ctrl.value for name,
                                ctrl in controls.items()}
                nodes, positions = _get_data(**control_args)
                source_all.data = dict(nodes=nodes, positions=positions)
            ctrl.on_change('value', update)

        doc.add_root(gridplot(
            [
                #[column(*controls.values())],
                [p2],
                [p1],
            ]
        ))

    handler = FunctionHandler(modify_doc)
    app = Application(handler)

    return app
