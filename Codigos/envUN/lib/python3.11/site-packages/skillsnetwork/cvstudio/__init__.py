import getpass
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import ibm_boto3
import requests
from ibm_botocore.client import Config
from tqdm import tqdm


def default(o):
    if isinstance(o, (date, datetime)):
        return o.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def isValidFolder():
    return bool(
        re.match(
            r"^/resources/labs/cvstudio/[0-9a-z-]+/(runs|apps)/[0-9a-z-]+$", os.getcwd()
        )
    )


def tokenToHeaders(token):
    return {"Authorization": "Bearer " + token, "Content-Type": "application/json"}


class CVStudio(object):
    @staticmethod
    def interactive(token=None, ibm_api_key_id=None, base_url=None):
        if base_url == None:
            base_url = os.environ.get(
                "CV_STUDIO_BASE_URL", "https://vision.skills.network"
            )

        if token is None:
            token = os.environ.get("CV_STUDIO_TOKEN")
        if token is None:
            # raise Exception('need to pass valid token or set CV_STUDIO_TOKEN environment variable')
            print(
                "Need to validate your identity as no CV Studio token set in environment"
            )
            print(
                "Please visit the following url in your browser and copy the token: "
                + base_url
                + "/api/token"
            )
            token = getpass.getpass("Enter your CV Studio token: ")

        projects = CVStudio.getProjects(token, base_url)

        print("Projects:")
        i = 1
        for project in projects:
            print(str(i) + ". " + project["name"])
            i += 1

        val = input("Choose a Project: ")
        val = int(val)

        project = projects[val - 1]
        print()
        print("You've chosen: " + project["name"])
        print()

        print("Connect to an Training Run or Application")
        print("1. Training Run")
        print("2. Application")

        val = input("Choose a type: ")

        if val == "1":
            print("You've chosen: Training Runs")
            print()

            runs = CVStudio.getRuns(token, base_url, project["id"])
            print("Training Runs in selected Project:")
            i = 1
            for run in runs:
                print(str(i) + ". " + run["name"])
                i += 1

            val = input("Choose a Training Run: ")
            val = int(val)

            run = runs[val - 1]
            print()
            print("You've chosen: " + run["name"])

            token = CVStudio.getTokenInteractive(token, base_url, "run", run["id"])

        elif val == "2":
            print("You've chosen: Applications")
            print()

            apps = CVStudio.getApplications(token, base_url, project["id"])
            print("Applications in selected Project:")
            i = 1
            for app in apps:
                print(str(i) + ". " + app["name"])
                i += 1

            val = input("Choose an Application: ")
            val = int(val)

            app = apps[val - 1]
            print()
            print("You've chosen: " + app["name"])

            token = CVStudio.getTokenInteractive(token, base_url, "app", app["id"])

        return token

    @staticmethod
    def getProjects(token, base_url):
        data = {"type": "projects"}
        x = requests.post(
            base_url + "/api/interactive",
            headers=tokenToHeaders(token),
            data=json.dumps(data, default=default),
        )
        if not x.ok:
            raise Exception("Failed to setup CV Studio client: {}".format(x.text))

        return json.loads(x.text)

    @staticmethod
    def getRuns(token, base_url, projectId):
        data = {"projectId": projectId, "type": "runs"}
        x = requests.post(
            base_url + "/api/interactive",
            headers=tokenToHeaders(token),
            data=json.dumps(data, default=default),
        )
        if not x.ok:
            raise Exception("Failed to setup CV Studio client")

        return json.loads(x.text)

    @staticmethod
    def getApplications(token, base_url, projectId):
        data = {"projectId": projectId, "type": "apps"}
        x = requests.post(
            base_url + "/api/interactive",
            headers=tokenToHeaders(token),
            data=json.dumps(data, default=default),
        )
        if not x.ok:
            raise Exception("Failed to setup CV Studio client")

        return json.loads(x.text)

    @staticmethod
    def getTokenInteractive(token, base_url, itemType, itemId):
        data = {"itemId": itemId, "type": itemType + "Token"}
        x = requests.post(
            base_url + "/api/interactive",
            headers=tokenToHeaders(token),
            data=json.dumps(data, default=default),
        )
        if not x.ok:
            raise Exception("Failed to setup CV Studio client")

        return json.loads(x.text)

    def __init__(
        self,
        interactive=False,
        token=None,
        ibm_api_key_id=None,
        base_url=None,
        public=False,
        settingsFile=".cvstudio.json",
    ):
        self.public = public

        if os.path.exists(settingsFile):
            with open(settingsFile) as f:
                settings = json.load(f)
            self.base_url = settings.get("base_url")
            self.token = settings.get("token")
            self.ibm_api_key_id = settings.get("ibmcloud_api_key")
            self.endpoint = settings.get("endpoint")
            self.bucket = settings.get("bucket")
            self.setup()
            return

        if base_url == None:
            self.base_url = os.environ.get(
                "CV_STUDIO_BASE_URL", "https://vision.skills.network"
            )
        else:
            self.base_url = base_url

        if interactive:
            result = CVStudio.interactive(base_url=base_url)
            if "token" in result:
                token = result["token"]
            if "apikey" in result:
                ibm_api_key_id = result["apikey"]

        if token == None:
            self.token = self.getToken()
            if self.token is None:
                raise Exception(
                    "need to pass valid token or set CV_STUDIO_TOKEN environment variable"
                )
        else:
            self.token = token

        if ibm_api_key_id == None:
            self.ibm_api_key_id = os.environ.get("IBMCLOUD_API_KEY")
            if self.ibm_api_key_id == None:
                raise Exception(
                    "need to pass valid ibm_api_key_id or set IBMCLOUD_API_KEY environment variable"
                )
        else:
            self.ibm_api_key_id = ibm_api_key_id

        self.setup()

    def getToken(self):
        token = os.environ.get("CV_STUDIO_TOKEN")
        if token is None:
            return None

        if not isValidFolder():
            raise Exception(
                "Need to run from a valid folder. e.g. /resources/labs/cvstudio/<project-name>/runs/<run-name>"
            )

        rest = os.getcwd()
        rest, itemName = os.path.split(rest)
        rest, itemType = os.path.split(rest)
        rest, projectName = os.path.split(rest)

        data = {"projectName": projectName, "itemType": itemType, "itemName": itemName}

        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }

        x = requests.post(
            self.base_url + "/api/token",
            headers=headers,
            data=json.dumps(data, default=default),
        )

        if not x.ok:
            raise Exception("Failed to setup CV Studio client")

        item = json.loads(x.text)
        return item["token"]

    def setup(self):
        if not "endpoint" in vars(self).keys():
            headers = {
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            }

            x = requests.post(self.base_url + "/api/cos_credentials", headers=headers)

            if x.ok:
                item = json.loads(x.text)

                self.endpoint = item["endpoint"]
                if self.public:
                    self.endpoint = self.endpoint.replace(".private", "")

                self.bucket = item["bucket"]
            else:
                print("Failed to setup CV Studio client")
                return x

        config = Config(
            connect_timeout=1, retries={"max_attempts": 2}, signature_version="oauth"
        )

        self.cos = ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.ibm_api_key_id,
            ibm_service_instance_id=self.ibm_api_key_id,
            config=config,
            endpoint_url=self.endpoint,
        )

        if self.public:
            return x

        try:
            self.cos.head_bucket(Bucket=self.bucket)
        except Exception as e:
            self.endpoint = self.endpoint.replace(".private", ".direct")

        self.cos = ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.ibm_api_key_id,
            ibm_service_instance_id=self.ibm_api_key_id,
            config=config,
            endpoint_url=self.endpoint,
        )

        try:
            self.cos.head_bucket(Bucket=self.bucket)
        except Exception as e:
            self.endpoint = self.endpoint.replace(".direct", "")

        self.cos = ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.ibm_api_key_id,
            ibm_service_instance_id=self.ibm_api_key_id,
            config=config,
            endpoint_url=self.endpoint,
        )

    def download_file_cos(self, key, folder=os.getcwd(), attempts=0):
        try:
            local_file_name = str(Path(folder).joinpath(key))
            res = self.cos.download_file(
                Bucket=self.bucket, Key=key, Filename=local_file_name
            )
        except Exception as e:
            print("Exception", e)
            if attempts < 3:
                print("Retrying download")
                self.download_file_cos(key, folder=folder, attempts=attempts + 1)

    def get_annotations(self):
        try:
            return json.loads(
                self.cos.get_object(Bucket=self.bucket, Key="_annotations.json")[
                    "Body"
                ].read()
            )
        except Exception as e:
            print("Exception", e)

    def downloadAll(self):
        image_folder = Path(os.getcwd()).joinpath("images")
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)

        for image in tqdm(self.get_annotations()["annotations"].keys()):
            self.download_file_cos(image, image_folder)

    def getDataset(
        self,
        transform=None,
        train_test=None,
        percentage_train=0.8,
        random_state=0,
        degrees=5,
    ):
        dataset = getDataset(
            annotations=self.get_annotations(),
            bucket_name=self.bucket,
            transform=transform,
            train_test=train_test,
            percentage_train=percentage_train,
            random_state=random_state,
            degrees=degrees,
        )
        return dataset

    def uploadModel(self, key, details={}):
        self.upload_file_cos(key)
        details["ready"] = True
        details["filename"] = key
        details["filesize"] = os.stat(key).st_size

        self.report(model=details)

    def downloadModel(self, folder=os.getcwd()):
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
        }

        x = requests.post(self.base_url + "/api/modelDetails", headers=headers)

        if x.ok:
            model = json.loads(x.text)

            if "external" in model:
                raise Exception("External models support coming soon.")
            elif "ready" not in model or not model["ready"]:
                raise Exception("Model isn't ready")

            self.download_file_cos(model["filename"], folder)

            return model

        else:
            raise Exception("Failed to get Model Details")

    def upload_file_cos(self, filename):
        try:
            res = self.cos.upload_file(
                filename, self.bucket, os.path.basename(filename)
            )
        except Exception as e:
            print("Exception", e)
        else:
            print("File Uploaded")

    def report(
        self,
        started=None,
        completed=None,
        url=None,
        parameters=None,
        accuracy=None,
        model=None,
    ):
        data = {}

        if started is not None:
            data["started"] = started

        if completed is not None:
            data["completed"] = completed

        if parameters is not None:
            data["parameters"] = parameters

        if accuracy is not None:
            data["accuracy"] = accuracy

        if model is not None:
            data["model"] = model

        if url is not None:
            data["url"] = url

        if len(data) == 0:
            raise Exception("Nothing to report")

        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
        }

        x = requests.post(
            self.base_url + "/api/report",
            headers=headers,
            data=json.dumps(data, default=default),
        )

        return x

    def ping(self):
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
        }

        x = requests.post(self.base_url + "/api/ping", headers=headers)

        return x


def getDataset(
    annotations,
    bucket_name,
    transform,
    train_test=None,
    percentage_train=0.8,
    random_state=0,
    degrees=5,
):
    import pandas as pd
    from PIL import Image
    from torch.utils.data import Dataset

    class Dataset(Dataset):
        def __init__(
            self,
            annotations,
            bucket_name,
            transform,
            train_test=None,
            percentage_train=percentage_train,
            random_state=random_state,
            degrees=degrees,
        ):

            self.train_test = train_test
            self.transform = transform
            if self.transform is None:
                print("defult transform for pretrained model resnet18")
                self.setDefaultTransform()

            self.train_test = train_test
            self.degrees = degrees

            problem_type = annotations["type"]

            if problem_type != "classification":
                raise Exception("Can only get Dataset of Classification models")

            labels_types = annotations["labels"]

            labels_filename = Path(os.getcwd()).joinpath(bucket_name + "_lables.csv")

            if os.path.exists(labels_filename):
                self.data = pd.read_csv(labels_filename)
            else:
                data_ = {"label": [], "y": [], "file_name": [], "key": []}
                for key, label_dict in annotations["annotations"].items():
                    filename = Path(os.getcwd()).joinpath("images", key)

                    if os.path.exists(filename):
                        label = label_dict[0]["label"]

                        data_["label"].append(label)
                        data_["y"].append(labels_types.index(label))
                        data_["key"].append(key)
                        data_["file_name"].append(filename)

                self.data = pd.DataFrame(data_)
                # train set
            if self.train_test == "train":
                print("this is the training set")
                self.data = self.data.sample(
                    frac=percentage_train, random_state=random_state
                ).reset_index()
                labels_filename = Path(os.getcwd()).joinpath(
                    bucket_name + "_train_lables.csv"
                )
                # self.data.to_csv(labels_filename)
            # test set
            if self.train_test == "test":
                print("this is the test set")
                labels_filename = Path(os.getcwd()).joinpath(
                    bucket_name + "_test_lables.csv"
                )
                temp = self.data.sample(
                    frac=percentage_train, random_state=random_state
                )
                self.data = self.data.drop(temp.index).reset_index()
                # self.data.to_csv(labels_filename)

            self.n_classes = len(self.data["y"].unique())

        def setDefaultTransform(self):
            from torchvision import transforms

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if self.train_test == "train":
                # Data augmentation and normalization for training
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                # Data augmentation and normalization for training
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            image = Image.open(self.data.loc[idx, "file_name"])
            y = self.data.loc[idx, "y"]
            if self.transform:
                image = self.transform(image)

            return image, y

    dataset = Dataset(
        annotations=annotations,
        bucket_name=bucket_name,
        transform=transform,
        train_test=train_test,
        percentage_train=percentage_train,
        random_state=random_state,
        degrees=degrees,
    )
    return dataset
