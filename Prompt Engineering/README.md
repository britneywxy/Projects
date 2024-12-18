# Prompt-Engineering
Prompt Engineering Final Project -- Calendar Planner Pro / Scheduling Assistant

This project aims to help user manage the time effectively. It integrates with the Google Calendar to understand the existing routines and uses ChatGPT's natural language processing capabilities to prioritize the new events and find avaiable time slots. Finally the optimized schedule, with the new events integrated into the available slots, is then pushed back and updated to the user's Google Calendar.

To run Planner Pro, user will need to set up authentication first. This involves:
- Creating a Google Cloud credential file `credential.json` from [Google Calendar API](https://developers.google.com/calendar/api/guides/overview)
- [Setting up the OpenAI API key](https://openai.com/blog/openai-api) and storing it as `OPENAI_API_KEY` varaible in the bash profile. 


After completing this setup, run the following code:
```python
python main.py
```
