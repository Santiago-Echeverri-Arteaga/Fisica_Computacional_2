import json
import argparse
import skillsnetwork.cvstudio


def upload(modelFilename, data, interactive):
    cvstudio = skillsnetwork.cvstudio.CVStudio(interactive=interactive)

    result = cvstudio.upload(**data)

    if result.ok:
        print("Congratulations your results have been reported back to CV Studio!")
    else:
        print("Failed to report results.")
        print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a Model to CV Studio")
    parser.add_argument(
        "-c",
        "--configFilename",
        help="Optional Filename of JSON files containing data to send to CV Studio",
    )
    parser.add_argument(
        "-i", "--interactive", nargs="*", help="Interactive login prompt"
    )
    parser.add_argument(
        "modelFilename",
        help="Filename of model file (e.g. model.pt) to send to CV Studio",
    )
    args = parser.parse_args()

    if args.configFilename:
        with open(args.configFilename) as f:
            data = json.load(f)

        upload(args.modelFilename, data, interactive=args.interactive is not None)
    else:
        upload(args.modelFilename, {}, interactive=args.interactive is not None)
