import json
import argparse
import skillsnetwork.cvstudio


def ping(interactive):
    cvstudio = skillsnetwork.cvstudio.CVStudio(interactive=interactive)

    result = cvstudio.ping()

    if result.ok:
        print("Congratulations you are connected to CV Studio!")
        item = json.loads(result.text)
        if "trainingrun" in item:
            print("Training Run: " + item["trainingrun"]["name"])
        elif "app" in item:
            print("Application: " + item["app"]["name"])
    else:
        print("Failed to contact CV Studio.")
        print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ping CV Studio")
    parser.add_argument(
        "-i", "--interactive", nargs="*", help="Interactive login prompt"
    )
    args = parser.parse_args()

    ping(interactive=args.interactive is not None)
