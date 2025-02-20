import argparse
import skillsnetwork.cvstudio


def setup(interactive):
    cvstudio = skillsnetwork.cvstudio.CVStudio(interactive=interactive)

    cvstudio.download_all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download all images from CV Studio project."
    )
    parser.add_argument(
        "-i", "--interactive", nargs="*", help="Interactive login prompt"
    )
    args = parser.parse_args()

    setup(interactive=args.interactive is not None)
