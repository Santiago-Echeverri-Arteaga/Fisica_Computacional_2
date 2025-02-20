import json
import argparse
import skillsnetwork.cvstudio


def download(interactive, token=None, public=False):
    cvstudio = skillsnetwork.cvstudio.CVStudio(
        interactive=interactive, token=token, public=public
    )

    model = cvstudio.downloadModel()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Model from CV Studio")
    parser.add_argument("-p", "--public", nargs="*", help="Access files publically")
    parser.add_argument(
        "-i", "--interactive", nargs="*", help="Interactive login prompt"
    )
    parser.add_argument("-t", "--token", nargs="*", help="CV Studio Token")
    parser.add_argument("-j", "--json", nargs="*", help="Output JSON")
    args = parser.parse_args()

    if args.token is not None:
        model = download(
            interactive=args.interactive is not None,
            public=args.public is not None,
            token=args.token[0],
        )
    else:
        model = download(
            interactive=args.interactive is not None, public=args.public is not None
        )

    if args.json is not None:
        print(json.dumps(model))
    else:
        print("Successfully download model to: " + model["filename"])
