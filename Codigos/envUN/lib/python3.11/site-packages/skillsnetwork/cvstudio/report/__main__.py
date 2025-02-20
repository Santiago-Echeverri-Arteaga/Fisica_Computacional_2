import json
import argparse
import skillsnetwork.cvstudio


def report(data, interactive):
    cvstudio = skillsnetwork.cvstudio.CVStudio(interactive=interactive)

    result = cvstudio.report(**data)

    if result.ok:
        print("Congratulations your results have been reported back to CV Studio!")
    else:
        print("Failed to report results.")
        print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a Report to CV Studio")
    parser.add_argument(
        "-i", "--interactive", nargs="*", help="Interactive login prompt"
    )
    parser.add_argument(
        "jsonFilename",
        nargs="+",
        help="Filename(s) of JSON files containing data to send to CV Studio",
    )
    args = parser.parse_args()

    data = {}
    for filename in args.jsonFilename:
        with open(filename) as f:
            data.update(json.load(f))

    report(data, interactive=args.interactive is not None)
