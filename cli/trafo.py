# get best global shift for transformation
import argparse

from decode import generic
from decode import io
from decode import neuralfitter
from decode import simulation


def main(
    path_trafo,
    img_size: tuple[int, int],
    mirror_dim: int | None,
    mirror_framesize: int | None,
    reference: str,
):
    """

    Args:
        path_trafo:
        img_size: image size
        mirror_dim: dimension to mirror
        mirror_framesize: size of the frame to infer mirroring from, which is likely
         different from the image size

    Returns:
        shift to be added to achieve global median shift close to zero
    """
    device = "cpu"
    trafo = io.trafo.load_xyz_trafo(
        path_trafo,
        scale=1 / 1000.0,
        switch_xy=True,
        shift=(1.0, 1.0, 0.0),
        reference=reference,
        device=device,
    )

    if mirror_dim is not None:
        t_mirr = simulation.trafo.pos.trafo.XYZMirrorAt.from_frame_flip(
            mirror_framesize, mirror_dim, device="cpu"
        )
        t_mirr = simulation.trafo.pos.trafo.XYZChanneledTransformation(t_mirr, ch=1)
        trafo.append(t_mirr)

    xextent, yextent = generic.utils.extent_by_sizes(img_size, px_size=(1, 1))
    indicator = neuralfitter.indicator.IndicatorChannelOffset(
        (xextent,), (yextent,), (img_size,), xy_trafo=trafo, device=device
    )
    offset_maps = indicator.forward()

    # calculate median offset
    offset = [-om.view(2, -1).median(1)[0].round().int() for om in offset_maps]

    return offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Best trafo offset")
    parser.add_argument("--path", type=str, required=True, help="Path to trafo")
    parser.add_argument(
        "--img_size", type=int, required=True, nargs=2, help="Image size"
    )
    parser.add_argument(
        "--mirror_dim", type=int, default=None, help="Dimension to mirror"
    )
    parser.add_argument(
        "--mirror_framesize",
        type=int,
        default=None,
        help="Frame size to infer mirroring from",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="trafo_raw",
        help="Reference to use for trafo, either 'trafo_raw' or 'trafo_inv_raw' "
        "which is necessary to load the correct direction of the trafo",
    )

    cli_args = parser.parse_args()

    offset = main(
        path_trafo=cli_args.path,
        img_size=cli_args.img_size,
        mirror_dim=cli_args.mirror_dim,
        mirror_framesize=cli_args.mirror_framesize,
        reference=cli_args.reference,
    )
    print(f"The best global offsets for the individual channels are")
    [print(f"ch {i}: {o.tolist()}") for i, o in enumerate(offset)]
    print(f"for an image size of {cli_args.img_size},"
          f"\nmirror dimension of {cli_args.mirror_dim}"
          f"\nand mirror frame size of {cli_args.mirror_framesize}")
