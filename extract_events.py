import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/tasks"]


def extract_events(utc_now):
  """
  input: utc_now (datetime object)
  return: event_extracted (a list of extracted events)

  Extracts existing routines/current schedules from the user's Google calendar based on the current time.
  If the current time is after 9 PM, it gets the events for the next day; otherwise, it gets the events for today.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json")
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)

    # Hours have passed for today
    hour_now = utc_now.hour
    # Decide on the time range based on the current time
    time_for_management = 21
    if hour_now >= time_for_management:  # After 9 PM, get the events for next day
        timeMin = (utc_now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        timeMax = (utc_now + datetime.timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        print("Getting Next Day's Events")
    else:  # Before 9 PM, get the events for today
        timeMin = utc_now.isoformat() + 'Z'
        timeMax = (utc_now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        print("Getting Today's Events")

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=timeMin,
            timeMax=timeMax,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
      print("No upcoming events found.")
      return

    event_extracted = []
    for event in events:
      start = event["start"].get("dateTime", event["start"].get("date"))
      end = event["end"].get("dateTime", event["end"].get("date"))
      event_extracted.append([event["summary"],start, end])
    return event_extracted
  except HttpError as error:
    print(f"An error occurred: {error}")


# if __name__ == "__main__":
#   extract_events()