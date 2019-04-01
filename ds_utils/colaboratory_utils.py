# Utils for Google Colaboratory (https://colab.research.google.com)

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
#!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build


def get_authenticated_drive_client():
    """
    Authenticate and create the PyDrive client.
    This only needs to be done once per notebook.
    :return:
    """
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive_client = GoogleDrive(gauth)
    return drive_client


def get_drive_service():
    """
    Create in order to export file to Drive
    :return:
    """
    drive_service = build('drive', 'v3')
    return drive_service


def import_file(drive_client, file_id, filepath):
    """
    Download a file based on its file ID.
    :param drive_client:
    :param file_id:
    :param filepath:
    :return:
    """
    downloaded = drive_client.CreateFile({'id': file_id})
    downloaded.GetContentFile(filepath)


# TODO check difference with export_file
def upload_file(drive_client, filename, filepath):
    """
    Upload file to Drive.
    :param drive_client:
    :return:
    """
    uploaded = drive_client.CreateFile(
        {'title': filename,
         'mimetype': 'text/plain'})
    uploaded.SetContentFile(filepath)
    uploaded.Upload()

    print('Uploaded file with ID {}'.format(uploaded.get('id')))


def export_file(drive_service, filename, filepath, folder_id=None):
    """
    Export a file to Drive
    :param folder_id: ID of the Drive folder where to insert the file
    :param drive_service:
    :param filepath:
    :param filename:
    :return:
    """
    file_metadata = {
        'name': filename
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(filepath,
                            mimetype='text/plain',
                            resumable=True)
    created = drive_service.files().create(body=file_metadata,
                                           media_body=media,
                                           fields='id').execute()

    print('Exported file with ID {}'.format(created.get('id')))
