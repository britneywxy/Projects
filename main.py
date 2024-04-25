import json
from datetime import datetime
import import_events
import extract_events
from arrange import arrange
import gui_window

def main():
    # User's current time
    utc_now = datetime.datetime.utcnow()

    # Extract the routines from user's Google Calendar
    routine = extract_events.extract_events(utc_now)
    print(routine)

    # Input the new/upcoming events that need to schedule in a GUI window
    input_text = gui_window.create_main_window()
    print(input_text)
    day_start = '08:30'
    day_end = '23:00'
    gmail = ''
    day_start = datetime.time(datetime.strptime(day_start, '%H:%M'))
    day_end = datetime.time(datetime.strptime(day_end, '%H:%M'))

    # Read the user input, utilize ChatGPT to prioritize the events
    # and find avaiable time slots
    arrange(input_text, routine, day_start, day_end, gmail)

    with open('events.json', 'r') as file:
        items = json.load(file)

    # Import and update the new events to user's Calendar
    for item in items:
        item.pop('istask', None)
        import_events.add_events(item)

if __name__ == '__main__':
    main()
