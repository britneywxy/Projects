import openai, os, json
from datetime import datetime, timedelta

## User API settings
api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(api_key=api_key)


def estimate_duration(event_name):
    """
    This function estimates the duration of an event in minutes, in case the user does not provide a duration.
    input: event_name (str), eg. 'Gym session'
    output: duration (float), in minutes
    """
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "Estimate the duration of the following event. \
             For example, output '60' for 1 hour, or '30' for 30 minutes. Only return an integer in minutes, no other text."},
            {"role": "user", "content": event_name},
        ],
    )
    result = response.choices[0].message.content
    return float(result)

def set_priority(prompt):
    """
    Sets the priority of the given events. 
    input: prompt (str), eg. 'Gym session, Grocery shopping'
    output: priority_list (list), eg. ['Gym session', 'Grocery shopping']
    """
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "In which order should I prioritize the following events? \
             Return the events in their original names and in descending order separated by commas. \
             For example, 'Event 1, Event 2, Event 3'"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    result = response.choices[0].message.content
    priority_list = [x.strip() for x in result.split(',') if x.strip() != '']
    return priority_list

def text2events(input_text):
    """
    Extracts the events and their durations from the input text.
    input: input_text (str), eg. "I have a course assignment due tomorrow 11pm"
    output: events and duration (dict) {'Event 1': timedelta(minutes=60), 'Event 2': timedelta(minutes=60), ...}
    """
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "Read from the following text and return the events and their durations IN MINUTES. \
             Follow exactly this format: 'Event 1, 60; Event 2, 30'. If no duration is specified, assume 0 minute."},
            {"role": "user", "content": input_text},
        ],
    )
    result = response.choices[0].message.content
    events = result.split(';')
    return {x.split(',')[0].strip(): timedelta(minutes=float(x.split(',')[1].strip())) for x in events}

def find_available_slots(routine_starts, routine_ends):
    """
    Find the available slots between the routine events.
    input: routine_starts (list), routine_ends (list)
    output: available_slots (list), eg. [['09:00', '10:00'], ['13:00', '14:00']]
    """
    available_slots = []
    for i in range(len(routine_starts)-1):
        if routine_ends[i] < routine_starts[i+1]:
            available_slots.append([routine_ends[i].time(), routine_starts[i+1].time()])
    return available_slots

def arrange_best_slot(event_name, event_duration, available_slots, target_date, tzinfo):
    """
    Arrange the best slot for the given event.
    input: event_name (str), event_duration (timedelta), available_slots (list), target_date (datetime.date), tzinfo (timezone)
    output: best_slot (list), eg. [datetime.datetime(...), datetime.datetime(...)]
    """
    available_slots = [f'{slot[0].strftime("%H:%M")}, {slot[1].strftime("%H:%M")}' for slot in available_slots]
    print(f'Available slots: {available_slots}')
    response = client.chat.completions.create(
        model = 'gpt-4-turbo',
        messages=[
            {"role": "system", "content": "Given the event name, the event duration, \
                and available slots, return a reasonable time slot to schedule the event. the event duration could be flexible. \
                Answer with EXACTLY the same format as this example: '09:00, 10:00' and do not output your reasoning.\
                If there are no available slots, return and only return 'none'."},
            {"role": "user", "content": f'The event name is {event_name}, the event duration is {event_duration},\
                and the available time slots are {available_slots}.'},
        ],
        temperature=0.7,
    )
    result = response.choices[0].message.content
    if result.lower() == 'none' or result.strip("'").lower() == 'none':
        return None
    else:
        output = [datetime.strptime(x.strip().strip("'"), '%H:%M').time() for x in result.split(',') if x.strip() != '']
        return [datetime.combine(target_date, x).replace(tzinfo=tzinfo) for x in output]

def arrange(input_text, routine, day_start, day_end, gmail):
    """
    The main function. Arranges the events in the input text based on the given routine.
    """
    event_dict = text2events(input_text) # {'Grocery shopping': datetime.timedelta(0), 'Gym session': datetime.timedelta(seconds=3600)}
    priority = set_priority(', '.join(event_dict.keys())) # ['Gym session', 'Grocery shopping']
    print('Priority list: ', priority)
    priority_is_done = {x: False for x in priority} # {'Gym session': False, 'Grocery shopping': False}

    # Adding DAY STARTS and DAY ENDS to the routine, so no events will be scheduled before or after the day
    # The idea:
    # routine_names = ['DAY STARTS', 'Event 1', 'Event 2', ..., 'Event N', 'DAY ENDS']
    # routine_starts = [00:00, Event 1 start, Event 2 start, ..., Event N start, DAY ENDS time]
    # routine_ends = [DAY STARTS time, Event 1 end, Event 2 end, ..., Event N end, 23:59]

    routine_names = [x[0] for x in routine] # ['Event 1', 'Event 2']
    routine_names.insert(0, 'DAY STARTS')
    routine_names.append('DAY ENDS')

    routine_starts = [x[1] for x in routine]
    routine_ends = [x[2] for x in routine]
    routine_starts = [datetime.fromisoformat(x) for x in routine_starts] # start time, datetime objects
    routine_ends = [datetime.fromisoformat(x) + timedelta(minutes=10) for x in routine_ends] # end time, datetime objects

    target_date = routine_starts[0].date() # get the date of the first event, this is where all events will be scheduled
    tzinfo = routine_starts[0].tzinfo # get the timezone info

    day_start_dt = datetime.combine(target_date, day_start).replace(tzinfo=tzinfo)
    day_end_dt = datetime.combine(target_date, day_end).replace(tzinfo=tzinfo)

    routine_starts.insert(0, datetime.combine(target_date, datetime.time(datetime.strptime('00:00', '%H:%M'))).replace(tzinfo=tzinfo))
    routine_starts.append(day_end_dt)

    routine_ends.insert(0, day_start_dt)
    routine_ends.append(datetime.combine(target_date, datetime.time(datetime.strptime('23:59', '%H:%M'))).replace(tzinfo=tzinfo))

    print('-'*50)

    # The new lists are created to store the new routine, which will be written to the JSON file
    new_routine_names = []
    new_routine_starts = []
    new_routine_ends = []

    for event in priority_is_done:
        if priority_is_done[event] == False:
            event_duration = event_dict[event]
            available_slots = find_available_slots(routine_starts, routine_ends)

            # If the event has no duration, estimate the duration
            if event_duration == timedelta(0):
                est_duration = min(estimate_duration(event), 180.)
                best_slot = arrange_best_slot(event, est_duration, available_slots, target_date, tzinfo)
            else:
                best_slot = arrange_best_slot(event, event_duration, available_slots, target_date, tzinfo)

            if best_slot == None:
                print(f'Could not find a suitable slot for {event}')
                continue
            else:
                priority_is_done[event] = True
                print(f'Best slot for {event}: {[i.strftime("%H:%M") for i in best_slot]}\n')
                event_start = best_slot[0]
                event_end = best_slot[1]

                # Update routine_starts and routine_ends
                routine_starts.append(event_start - timedelta(minutes=10)) # - 10 minutes to ensure 10 min gap before the fixed routines
                routine_ends.append(event_end + timedelta(minutes=10)) # + 10 minutes to ensure 10 min gap after the arranged slot of a given event
                routine_names.append(event)
                new_routine_starts.append(event_start)
                new_routine_ends.append(event_end)
                new_routine_names.append(event)

                # Sort the arrays based on start time
                routine_starts, routine_ends, routine_names = \
                    [list(x) for x in zip(*sorted(zip(routine_starts, routine_ends, routine_names)))]
                new_routine_starts, new_routine_ends, new_routine_names = \
                    [list(x) for x in zip(*sorted(zip(new_routine_starts, new_routine_ends, new_routine_names)))]


    # Create a list to store all the events
    events = []

    for new_event in zip(new_routine_names, new_routine_starts, new_routine_ends):
        # if new_event[1] == new_event[2]:
        #     istask = True
        # else:
        #     istask = False
        event = {
            # "istask": istask,
            "summary": new_event[0],
            # "description": "testing the calendar project",
            "colorId": 1,
            "start": {
                "dateTime": new_event[1].isoformat(),
                "timeZone": "(GMT-04:00) Eastern Time - New York"
            },
            "end": {
                "dateTime": new_event[2].isoformat(),
                "timeZone": "(GMT-04:00) Eastern Time - New York"
            },
            "attendees": [{"email": gmail}]
        }

        events.append(event)

    # Write the events to a JSON file
    with open("events.json", "w") as file:
        json.dump(events, file, indent=4)


# For test purposes
if __name__ == '__main__':

    ## User settings
    input_text = "I have a course assignment due tomorrow 11pm, preparation for tomorrow's exam at 11pm, project meeting, celebrating best friend's birthday, gym session for 1 hour, and grocery shopping"
    routine = [['Mini Course', '2024-04-22T09:30:00-04:00', '2024-04-22T10:50:00-04:00'], ['Semester Course', '2024-04-22T11:00:00-04:00', '2024-04-22T12:20:00-04:00']]
    day_start = '08:30'
    day_end = '23:00'
    gmail = ""
    day_start = datetime.time(datetime.strptime(day_start, '%H:%M'))
    day_end = datetime.time(datetime.strptime(day_end, '%H:%M'))

    arrange(input_text, routine, day_start, day_end, gmail)
