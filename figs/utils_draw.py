from quimb.schematic import Drawing, get_color


color_tensor = get_color("blue")
color_tensor_su = get_color("bluedark")
color_gauge = get_color("pink")
color_su_gauge = (0.7, 0.5, 0.8)
# color_su_gauge_sqrt = (0.5, 0.9, 0.3)
color_su_gauge_sqrt = (0.59, 0.73, 0.34)
color_op = get_color("orange")


def get_axes_area_fraction(ax):
    bbox = ax.get_position()
    axes_area = bbox.width * bbox.height
    return axes_area


def get_presets(lw_scale=1.0, r_scale=1.0):

    hatch_gauges = "......"
    hatch_target = "/////"
    patch_color = (0.5, 0.5, 0.5)  # get_color("blue")
    patch_edgecolor = "black"
    r = 0.25 * r_scale

    return {
        "bond": {
            "linewidth": 2 * lw_scale,
            "shorten": r,
        },
        "phys": {
            "linewidth": lw_scale,
            "shorten": 2 / 3 * r,
            "width": 0.015,
        },
        "tensor": {
            "linewidth": 2 * lw_scale,
            "color": color_tensor,
            # "edgecolor": "black",
            "radius": r,
        },
        "tensor_su": {
            "linewidth": 2 * lw_scale,
            "color": color_tensor_su,
            # "edgecolor": "black",
            "radius": r,
        },
        "target": {
            "linewidth": 2 * lw_scale,
            "color": color_op,
            # "edgecolor": "black",
            "radius": r,
            "hatch": hatch_target,
        },
        "gauge": {
            "radius": r / 3,
            "linewidth": 2 * lw_scale,
            "hatch": hatch_gauges,
            "color": color_gauge,
            "zorder_delta": 0.15,
            "marker": "o",
        },
        "gauge_su": {
            "radius": r / 3,
            "linewidth": 2 * lw_scale,
            "hatch": hatch_gauges,
            "color": color_su_gauge,
            "zorder_delta": 0.15,
            "marker": "s",
        },
        "gauge_su_sqrt": {
            "radius": r / 3,
            "linewidth": 2 * lw_scale,
            "hatch": hatch_gauges,
            "color": color_su_gauge_sqrt,
            "zorder_delta": 0.15,
            "marker": "^",
        },
        "cluster": {
            "radius": 0.4,
            "facecolor": patch_color,
            "alpha": 0.2,
            "linewidth": 1,
            "linestyle": ":",
            "edgecolor": patch_edgecolor,
        },
    }


def draw_cluster(
    cluster,
    targets=(),
    d=None,
    highlight=(),
    leave_open=(),
    xshift=0,
    yshift=0,
    labels={},
    label_fontsize=12,
    doffset=0.04,
):

    if d is None:
        d = Drawing(presets=get_presets())

    def shift_point(x, y):
        return (x + xshift, y + yshift)

    sites = set(cluster)
    bonds = {}
    gauges = {}
    shortens = {}

    for cooa in cluster:
        x, y = cooa
        for coob in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            if coob in sites:
                bonds[cooa, coob] = None
                shortens[cooa, coob] = d.presets["tensor"]["radius"]
            else:
                coog = ((x + coob[0]) / 2, (y + coob[1]) / 2)
                gauges[coog] = None
                bonds[cooa, coog] = None
                shortens[cooa, coog] = (
                    d.presets["tensor"]["radius"],
                    0.0,
                )

    # draw bonds
    for cooa, coob in bonds:
        xa, ya = cooa
        xb, yb = coob
        if xa == xb:
            offsetx = doffset
        else:
            offsetx = 0.0
        if ya == yb:
            offsety = doffset
        else:
            offsety = 0.0

        shorten = shortens[cooa, coob]

        d.line(
            shift_point(xa - offsetx, ya - offsety),
            shift_point(xb - offsetx, yb - offsety),
            preset="bond",
            zorder=-11,
            shorten=shorten,
        )
        d.line(
            shift_point(xa + offsetx, ya + offsety),
            shift_point(xb + offsetx, yb + offsety),
            preset="bond",
            zorder=9,
            shorten=shorten,
        )
        if {cooa, coob} == set(targets):
            xa, ya = cooa
            xb, yb = coob
            d.line(
                shift_point(xa, ya),
                shift_point(xb, yb),
                preset="bond",
                color=get_color("red"),
                linewidth=1.5,
            )

    for coog in gauges:
        xg, yg = coog
        if (xg, yg) in leave_open:
            continue
        d.circle(shift_point(xg, yg), preset="gauge", zorder=10)

    # draw site tensors
    for i, j in sites:
        if (i, j) in targets:
            preset = "target"
        else:
            preset = "tensor"
        # d.circle((i, j), preset=preset)
        d.circle(
            shift_point(i - doffset / 2, j - doffset / 2),
            preset=preset,
            zorder=-10,
        )
        d.circle(
            shift_point(i + doffset / 2, j + doffset / 2),
            preset=preset,
            zorder=+10,
        )

    if highlight:
        for region in highlight:
            d.patch_around(
                tuple((x + xshift, y + yshift) for x, y in region),
                preset="cluster",
                zorder=-100,
            )

    for coo, label in labels.items():
        x, y = coo
        d.text(
            shift_point(x + 0.3, y - 0.3),
            label,
            zorder=20,
            fontsize=label_fontsize,
        )
        # d.text(shift_point(x, y), label, zorder=20, color="white")


def traced_bond(d, cooa, coob, drawtype=""):
    x, y = cooa
    xn, yn = coob
    e = 0.5
    xm = (1 - e) * x + e * xn
    ym = (1 - e) * y + e * yn
    bezier_pts = [
        (x, y, 1),
        (xm, ym, 1),
        (xm, ym, 1),
        (xm, ym, 0.5),
        (xm, ym, 0),
        (xm, ym, 0),
        (x, y, 0),
    ]
    d.bezier(bezier_pts, preset="bond")

    if drawtype == "su":
        d.square((xm, ym, 0.5 + 0.15), preset="gauge_su", zorder_delta=0.5)
        d.square((xm, ym, 0.5 - 0.15), preset="gauge_su", zorder_delta=0.5)
    elif drawtype == "bp":
        d.circle((xm, ym, 0.5), preset="gauge")


def draw_cluster_3d(cluster, drawtype="su", targets=(), d=None):
    """Draw a full 3d cluster."""
    if d is None:
        d = Drawing(
            presets=get_presets(),
            a=60,
            b=10,
            xscale=1,
            yscale=1.2,
            zscale=0.6,
        )

    boundary = {}
    bonds = set()

    if drawtype == "su":
        tensor_preset = "tensor_su"
        gauge_preset = "gauge_su"
    else:
        tensor_preset = "tensor"
        gauge_preset = "gauge"

    for coo in cluster:
        i, j = coo
        d.circle((i, j, 0), preset=tensor_preset)
        d.zigzag((i, j, 0), (i, j, 1), preset="phys")
        d.circle((i, j, 1), preset=tensor_preset)

        for coon in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
            if coon not in cluster:
                boundary.setdefault(coon, []).append((i, j))
            else:
                pair = tuple(sorted([(i, j), coon]))
                bonds.add(pair)

    for cooa, coob in bonds:
        # draw all bonds
        d.line((*cooa, 0), (*coob, 0), preset="bond")
        d.line((*cooa, 1), (*coob, 1), preset="bond")
        xm = cooa[0] / 2 + coob[0] / 2
        ym = cooa[1] / 2 + coob[1] / 2

        if drawtype == "su":
            # draw gauge on bonds
            d.square((xm, ym, 0), preset=gauge_preset, zorder_delta=0.5)
            d.square((xm, ym, 1), preset=gauge_preset, zorder_delta=0.5)

        if set([cooa, coob]) == set(targets):
            h = 0.25
            ol = 1.05

            (xa, ya), (xb, yb) = targets
            xga = ol * xa + (1 - ol) * xb
            yga = ol * ya + (1 - ol) * yb
            xgb = ol * xb + (1 - ol) * xa
            ygb = ol * yb + (1 - ol) * ya

            d.shape(
                [
                    (xga, yga, 0.5 - h / 2),
                    (xga, yga, 0.5 + h / 2),
                    (xgb, ygb, 0.5 + h / 2),
                    (xgb, ygb, 0.5 - h / 2),
                ],
                color=d.presets["target"]["color"],
                hatch=d.presets["target"]["hatch"],
                linewidth=d.presets["target"]["linewidth"],
                zorder_aggregate="max",
                # zorder_delta=0.5,
            )

    for coon, coos in boundary.items():
        # draw boundary contraction and messages
        xn, yn = coon
        for coo in coos:
            traced_bond(d, coo, coon, drawtype)
