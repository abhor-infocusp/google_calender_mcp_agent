from string import Template

# Decide the person
persona_prompt = Template("""
For the profession "$profession", list a number of different instances of this profession.
For example, a "teacher" can kindergarden teacher, high school, university, physics, maths, economics teacher etc.
Select one of these instances randomly.

Then:
1. Print "##################"
2. Create a persona of a person in this profession: short description of this person's professional or personal life. 
3. Write their role, the typical responsibilities, daily or weekly routines, and the main challenges they face. 
4. List people that are relevant based on the person's description - separate them for different purposes: team, work, personal, specific chaellenge etc.
Keep it concise, clear, and realistic.
Only output what is asked for - no extra fluff.
""")

# Example output
'''Profile: Senior Hospital Administrator
Role: A high-level executive responsible for overseeing the daily operations, financial health, and regulatory compliance of a mid-sized regional hospital.

Typical Responsibilities:

Managing multi-million dollar departmental budgets and resource allocation.

Ensuring the facility meets strict healthcare safety and legal standards.

Bridging communication between the Board of Directors and medical staff.

Implementing strategic initiatives to improve patient care outcomes.

Daily/Weekly Routine:

Morning: Rounds through various departments to assess facility conditions and staff morale; reviewing critical incident reports from the overnight shift.

Mid-day: Back-to-back meetings with department heads (ER, Surgery, HR) to address staffing shortages or equipment needs.

Afternoon: Reviewing financial forecasts and negotiating contracts with insurance providers or medical supply vendors.

Weekly: Formal board presentations and community outreach meetings.

Main Challenges:

Staff Burnout: Addressing the chronic shortage of nursing staff and high turnover rates.

Budgetary Constraints: Balancing the rising costs of medical technology with the need for affordable patient care.

Crisis Management: Responding to sudden surges in patient volume or unexpected regulatory audits.

Relevant Contacts
Core Team (Operations)

Chief Nursing Officer (CNO): For clinical staffing and bedside care standards.

Chief Financial Officer (CFO): For budget approvals and revenue cycle management.

HR Director: For recruitment strategies and labor union negotiations.

External Work Partners

Medical Supply Vendors: For procuring essential PPE and surgical equipment.

Government Health Inspectors: For maintaining hospital accreditation.

Insurance Executives: For reimbursement rate discussions.

Personal Support

Executive Coach: To manage high-stakes decision-making stress.

Immediate Family: For emotional decompression and work-life boundaries.

Specific Challenge: Regulatory Audit

Compliance Officer: To ensure all documentation is ready for inspection.

Legal Counsel: To interpret complex healthcare mandates and liability risks.'''

# Design the calender
architect_prompt = Template('''
You are a synthetic data planner. Generate a Calendar Blueprint for the given persona.
Persona:
$persona

Step 1: Define density (how many events per day? Eg. 3 work and 2 personal events on weekdays and 3 personal events on weekends)
Step 2: Create a Text-Based Schedule for Monday through Sunday, along with emails of any attendees. Don't add attendees if it doesn't make sense.
Format: 
  start_time - end_time: Event name (attendees: email, email, ...)

Make sure it is realistic and adheres to the selected density. 
''')

# Example output
'''As a synthetic data planner, I have designed this calendar blueprint to reflect the high-pressure, regulatory-heavy environment of a Senior Hospital Administrator.

Step 1: Density Definition
Weekdays: 6 Work Events, 2 Personal Events (reflecting a high-intensity executive role).

Weekends: 3 Personal/Maintenance Events.

Tone: High-stakes, focused on crisis management, financial oversight, and staff retention.

Step 2: Calendar Blueprint
Monday: Operations & Crisis Review
06:30 - 07:30: Personal: Morning HIIT and Mental Prep

08:00 - 09:15: Morning Rounds & Overnight Incident Review (attendees: cno@regionalhealth.org)

10:00 - 11:30: Emergency Department Staffing Emergency (attendees: cno@regionalhealth.org, hr_director@regionalhealth.org)

12:00 - 13:00: Working Lunch: Budget Reallocation Draft (attendees: cfo@regionalhealth.org)

13:30 - 15:00: Vendor Negotiation: MRI Maintenance Contracts (attendees: sales_lead@medtechsolutions.com)

15:30 - 16:30: Legal Briefing: Upcoming Regulatory Audit (attendees: compliance@regionalhealth.org, legal@healthcarelaw.com)

17:00 - 18:00: 1:1 Executive Coaching Session (attendees: coach.sarah@executive-growth.com)

19:00 - 20:30: Personal: Family Dinner (No Devices)

Tuesday: Strategic Planning & Compliance
07:00 - 08:00: Personal: Meditation and Breakfast

08:30 - 10:00: Facility Walkthrough: Safety Standards Audit (attendees: compliance@regionalhealth.org)

10:30 - 12:00: Recruitment Strategy: Nursing Retention Taskforce (attendees: hr_director@regionalhealth.org, cno@regionalhealth.org)

13:00 - 14:30: Insurance Reimbursement Rate Review (attendees: regional_vp@bluecross-example.com, cfo@regionalhealth.org)

15:00 - 16:00: Board of Directors Prep: Quarterly Financials

16:30 - 17:30: Community Health Initiative Workshop (attendees: mayor_liaison@cityhall.gov)

18:00 - 19:00: Reviewing Legal Liability Documentation

19:30 - 21:00: Personal: Reading (Non-Work Related)

Wednesday: Financial Health & Vendor Relations
06:00 - 07:30: Personal: Swimming Laps

08:30 - 09:30: Critical Incident Debrief (attendees: cno@regionalhealth.org, legal@healthcarelaw.com)

10:00 - 11:30: Medical Supply Chain Strategy: PPE Shortfalls (attendees: logistics@globalsupply.com, cfo@regionalhealth.org)

12:00 - 13:30: HR Session: Labor Union Negotiation Prep (attendees: hr_director@regionalhealth.org)

14:00 - 15:30: Capital Expenditure Planning: Surgical Wing (attendees: cfo@regionalhealth.org)

16:00 - 17:00: Physician Leadership Council Monthly (attendees: chief_of_staff@regionalhealth.org)

17:30 - 18:30: Executive Coach: Conflict Resolution Techniques (attendees: coach.sarah@executive-growth.com)

19:30 - 21:00: Personal: Evening Walk with Spouse

Thursday: Regulatory Focus & Board Reporting
07:30 - 08:30: Personal: Yoga

09:00 - 11:00: Mock Audit: Compliance & Documentation (attendees: compliance@regionalhealth.org, state_inspector_lead@healthdept.gov)

11:30 - 12:30: Department Head Sync: ER & Surgery Sync (attendees: surgery_lead@regionalhealth.org, er_director@regionalhealth.org)

13:30 - 15:30: Formal Board of Directors Presentation (attendees: board_chair@regionalhealth.org, cfo@regionalhealth.org)

16:00 - 17:00: Strategic Initiative: Patient Outcome Metrics (attendees: cno@regionalhealth.org)

17:15 - 18:15: New Hire Orientation (Keynote Speech)

18:30 - 19:30: Reviewing Overnight On-Call Schedule

20:00 - 21:30: Personal: Movie Night with Kids

Friday: Talent & Long-term Strategy
07:00 - 08:00: Personal: Morning Run

08:30 - 10:00: Morning Rounds: Staff Morale Focus (attendees: hr_director@regionalhealth.org)

10:30 - 12:00: Final Review: Regulatory Audit Submission (attendees: compliance@regionalhealth.org, legal@healthcarelaw.com)

12:30 - 14:00: Lunch Meeting: Philanthropic Donor Prospect (attendees: foundation_director@regionalhealth.org)

14:30 - 15:30: Financial Forecast Q3 Review (attendees: cfo@regionalhealth.org)

16:00 - 17:00: Weekly Wrap-Up: Executive Leadership Team (attendees: cfo@regionalhealth.org, cno@regionalhealth.org, hr_director@regionalhealth.org)

17:30 - 18:30: Inbox Zero & Weekend Prep

19:00 - 21:30: Personal: Dinner at Bistro with Friends

Saturday: Recovery & Family
09:00 - 10:30: Personal: Yard Work / Gardening

11:30 - 13:30: Personal: Kids’ Soccer Match

18:00 - 20:00: Personal: Weekly Grocery Planning & Meal Prep

Sunday: Reflection & Setup
10:00 - 12:00: Personal: Extended Family Brunch

14:00 - 16:00: Personal: Rest & Reflection

20:00 - 21:00: Personal: Calendar Review for Upcoming Audit Week

Would you like me to generate a specific email thread regarding the "Emergency Department Staffing" meeting mentioned on Monday?'''

# Jsonize - only for conver
jsonizer_prompt = Template('''
Convert the provided Calendar Blueprint into a strictly valid JSON object.
You are given the following list of events:
$calender_text

The week starts on Monday $monday_date. Assign dates sequentially: Monday=$monday_date, Tuesday is the next day, Wednesday the day after, and so on.

Constraints:
1. Root: Object with days as keys ("Monday", "Tuesday", etc.).
2. Start/End: Direct strings in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).
3. Attendees: Must be an Array of Strings. If empty, return [].
4. No Prose: Do not include any text other than the JSON block.

Example (if Monday were 2025-03-10):
{
  "Monday": [
    {
      "summary": "Project Sync",
      "start": "2025-03-10T10:00:00",
      "end": "2025-03-10T11:00:00",
      "attendees": ["dev@company.com", "pm@company.com"]
    },
    {
      "summary": "Deep Work",
      "start": "2025-03-10T13:00:00",
      "end": "2025-03-10T15:00:00",
      "attendees": []
    }
  ],
  "Tuesday": [
    {
      "summary": "Standup",
      "start": "2025-03-11T09:00:00",
      "end": "2025-03-11T09:30:00",
      "attendees": []
    }
  ]
}
''')

# Example output
'''I have converted your Calendar Blueprint into a strictly valid JSON object. For the purpose of the `ISO_8601` timestamps, I have used a placeholder date of **January 12th through January 18th, 2026**, assuming a standard Monday-start week.

```json
{
  "calendar_events": [
    {
      "summary": "Personal: Morning HIIT and Mental Prep",
      "start": { "dateTime": "2026-01-12T06:30:00" },
      "end": { "dateTime": "2026-01-12T07:30:00" },
      "attendees": ""
    },
    {
      "summary": "Morning Rounds & Overnight Incident Review",
      "start": { "dateTime": "2026-01-12T08:00:00" },
      "end": { "dateTime": "2026-01-12T09:15:00" },
      "attendees": "cno@regionalhealth.org"
    },
    {
      "summary": "Emergency Department Staffing Emergency",
      "start": { "dateTime": "2026-01-12T10:00:00" },
      "end": { "dateTime": "2026-01-12T11:30:00" },
      "attendees": "cno@regionalhealth.org, hr_director@regionalhealth.org"
    },
    {
      "summary": "Working Lunch: Budget Reallocation Draft",
      "start": { "dateTime": "2026-01-12T12:00:00" },
      "end": { "dateTime": "2026-01-12T13:00:00" },
      "attendees": "cfo@regionalhealth.org"
    },
    {
      "summary": "Vendor Negotiation: MRI Maintenance Contracts",
      "start": { "dateTime": "2026-01-12T13:30:00" },
      "end": { "dateTime": "2026-01-12T15:00:00" },
      "attendees": "sales_lead@medtechsolutions.com"
    },
    {
      "summary": "Legal Briefing: Upcoming Regulatory Audit",
      "start": { "dateTime": "2026-01-12T15:30:00" },
      "end": { "dateTime": "2026-01-12T16:30:00" },
      "attendees": "compliance@regionalhealth.org, legal@healthcarelaw.com"
    },
    {
      "summary": "1:1 Executive Coaching Session",
      "start": { "dateTime": "2026-01-12T17:00:00" },
      "end": { "dateTime": "2026-01-12T18:00:00" },
      "attendees": "coach.sarah@executive-growth.com"
    },
    {
      "summary": "Personal: Family Dinner (No Devices)",
      "start": { "dateTime": "2026-01-12T19:00:00" },
      "end": { "dateTime": "2026-01-12T20:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Meditation and Breakfast",
      "start": { "dateTime": "2026-01-13T07:00:00" },
      "end": { "dateTime": "2026-01-13T08:00:00" },
      "attendees": ""
    },
    {
      "summary": "Facility Walkthrough: Safety Standards Audit",
      "start": { "dateTime": "2026-01-13T08:30:00" },
      "end": { "dateTime": "2026-01-13T10:00:00" },
      "attendees": "compliance@regionalhealth.org"
    },
    {
      "summary": "Recruitment Strategy: Nursing Retention Taskforce",
      "start": { "dateTime": "2026-01-13T10:30:00" },
      "end": { "dateTime": "2026-01-13T12:00:00" },
      "attendees": "hr_director@regionalhealth.org, cno@regionalhealth.org"
    },
    {
      "summary": "Insurance Reimbursement Rate Review",
      "start": { "dateTime": "2026-01-13T13:00:00" },
      "end": { "dateTime": "2026-01-13T14:30:00" },
      "attendees": "regional_vp@bluecross-example.com, cfo@regionalhealth.org"
    },
    {
      "summary": "Board of Directors Prep: Quarterly Financials",
      "start": { "dateTime": "2026-01-13T15:00:00" },
      "end": { "dateTime": "2026-01-13T16:00:00" },
      "attendees": ""
    },
    {
      "summary": "Community Health Initiative Workshop",
      "start": { "dateTime": "2026-01-13T16:30:00" },
      "end": { "dateTime": "2026-01-13T17:30:00" },
      "attendees": "mayor_liaison@cityhall.gov"
    },
    {
      "summary": "Reviewing Legal Liability Documentation",
      "start": { "dateTime": "2026-01-13T18:00:00" },
      "end": { "dateTime": "2026-01-13T19:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Reading (Non-Work Related)",
      "start": { "dateTime": "2026-01-13T19:30:00" },
      "end": { "dateTime": "2026-01-13T21:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Swimming Laps",
      "start": { "dateTime": "2026-01-14T06:00:00" },
      "end": { "dateTime": "2026-01-14T07:30:00" },
      "attendees": ""
    },
    {
      "summary": "Critical Incident Debrief",
      "start": { "dateTime": "2026-01-14T08:30:00" },
      "end": { "dateTime": "2026-01-14T09:30:00" },
      "attendees": "cno@regionalhealth.org, legal@healthcarelaw.com"
    },
    {
      "summary": "Medical Supply Chain Strategy: PPE Shortfalls",
      "start": { "dateTime": "2026-01-14T10:00:00" },
      "end": { "dateTime": "2026-01-14T11:30:00" },
      "attendees": "logistics@globalsupply.com, cfo@regionalhealth.org"
    },
    {
      "summary": "HR Session: Labor Union Negotiation Prep",
      "start": { "dateTime": "2026-01-14T12:00:00" },
      "end": { "dateTime": "2026-01-14T13:30:00" },
      "attendees": "hr_director@regionalhealth.org"
    },
    {
      "summary": "Capital Expenditure Planning: Surgical Wing",
      "start": { "dateTime": "2026-01-14T14:00:00" },
      "end": { "dateTime": "2026-01-14T15:30:00" },
      "attendees": "cfo@regionalhealth.org"
    },
    {
      "summary": "Physician Leadership Council Monthly",
      "start": { "dateTime": "2026-01-14T16:00:00" },
      "end": { "dateTime": "2026-01-14T17:00:00" },
      "attendees": "chief_of_staff@regionalhealth.org"
    },
    {
      "summary": "Executive Coach: Conflict Resolution Techniques",
      "start": { "dateTime": "2026-01-14T17:30:00" },
      "end": { "dateTime": "2026-01-14T18:30:00" },
      "attendees": "coach.sarah@executive-growth.com"
    },
    {
      "summary": "Personal: Evening Walk with Spouse",
      "start": { "dateTime": "2026-01-14T19:30:00" },
      "end": { "dateTime": "2026-01-14T21:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Yoga",
      "start": { "dateTime": "2026-01-15T07:30:00" },
      "end": { "dateTime": "2026-01-15T08:30:00" },
      "attendees": ""
    },
    {
      "summary": "Mock Audit: Compliance & Documentation",
      "start": { "dateTime": "2026-01-15T09:00:00" },
      "end": { "dateTime": "2026-01-15T11:00:00" },
      "attendees": "compliance@regionalhealth.org, state_inspector_lead@healthdept.gov"
    },
    {
      "summary": "Department Head Sync: ER & Surgery Sync",
      "start": { "dateTime": "2026-01-15T11:30:00" },
      "end": { "dateTime": "2026-01-15T12:30:00" },
      "attendees": "surgery_lead@regionalhealth.org, er_director@regionalhealth.org"
    },
    {
      "summary": "Formal Board of Directors Presentation",
      "start": { "dateTime": "2026-01-15T13:30:00" },
      "end": { "dateTime": "2026-01-15T15:30:00" },
      "attendees": "board_chair@regionalhealth.org, cfo@regionalhealth.org"
    },
    {
      "summary": "Strategic Initiative: Patient Outcome Metrics",
      "start": { "dateTime": "2026-01-15T16:00:00" },
      "end": { "dateTime": "2026-01-15T17:00:00" },
      "attendees": "cno@regionalhealth.org"
    },
    {
      "summary": "New Hire Orientation (Keynote Speech)",
      "start": { "dateTime": "2026-01-15T17:15:00" },
      "end": { "dateTime": "2026-01-15T18:15:00" },
      "attendees": ""
    },
    {
      "summary": "Reviewing Overnight On-Call Schedule",
      "start": { "dateTime": "2026-01-15T18:30:00" },
      "end": { "dateTime": "2026-01-15T19:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Movie Night with Kids",
      "start": { "dateTime": "2026-01-15T20:00:00" },
      "end": { "dateTime": "2026-01-15T21:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Morning Run",
      "start": { "dateTime": "2026-01-16T07:00:00" },
      "end": { "dateTime": "2026-01-16T08:00:00" },
      "attendees": ""
    },
    {
      "summary": "Morning Rounds: Staff Morale Focus",
      "start": { "dateTime": "2026-01-16T08:30:00" },
      "end": { "dateTime": "2026-01-16T10:00:00" },
      "attendees": "hr_director@regionalhealth.org"
    },
    {
      "summary": "Final Review: Regulatory Audit Submission",
      "start": { "dateTime": "2026-01-16T10:30:00" },
      "end": { "dateTime": "2026-01-16T12:00:00" },
      "attendees": "compliance@regionalhealth.org, legal@healthcarelaw.com"
    },
    {
      "summary": "Lunch Meeting: Philanthropic Donor Prospect",
      "start": { "dateTime": "2026-01-16T12:30:00" },
      "end": { "dateTime": "2026-01-16T14:00:00" },
      "attendees": "foundation_director@regionalhealth.org"
    },
    {
      "summary": "Financial Forecast Q3 Review",
      "start": { "dateTime": "2026-01-16T14:30:00" },
      "end": { "dateTime": "2026-01-16T15:30:00" },
      "attendees": "cfo@regionalhealth.org"
    },
    {
      "summary": "Weekly Wrap-Up: Executive Leadership Team",
      "start": { "dateTime": "2026-01-16T16:00:00" },
      "end": { "dateTime": "2026-01-16T17:00:00" },
      "attendees": "cfo@regionalhealth.org, cno@regionalhealth.org, hr_director@regionalhealth.org"
    },
    {
      "summary": "Inbox Zero & Weekend Prep",
      "start": { "dateTime": "2026-01-16T17:30:00" },
      "end": { "dateTime": "2026-01-16T18:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Dinner at Bistro with Friends",
      "start": { "dateTime": "2026-01-16T19:00:00" },
      "end": { "dateTime": "2026-01-16T21:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Yard Work / Gardening",
      "start": { "dateTime": "2026-01-17T09:00:00" },
      "end": { "dateTime": "2026-01-17T10:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Kids’ Soccer Match",
      "start": { "dateTime": "2026-01-17T11:30:00" },
      "end": { "dateTime": "2026-01-17T13:30:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Weekly Grocery Planning & Meal Prep",
      "start": { "dateTime": "2026-01-17T18:00:00" },
      "end": { "dateTime": "2026-01-17T20:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Extended Family Brunch",
      "start": { "dateTime": "2026-01-18T10:00:00" },
      "end": { "dateTime": "2026-01-18T12:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Rest & Reflection",
      "start": { "dateTime": "2026-01-18T14:00:00" },
      "end": { "dateTime": "2026-01-18T16:00:00" },
      "attendees": ""
    },
    {
      "summary": "Personal: Calendar Review for Upcoming Audit Week",
      "start": { "dateTime": "2026-01-18T20:00:00" },
      "end": { "dateTime": "2026-01-18T21:00:00" },
      "attendees": ""
    }
  ]
}

```

---

**Next Step:** Would you like me to generate that specific email thread regarding the **"Emergency Department Staffing"** meeting for Monday?'''

# Generate query
query_prompt = Template('''
$calender_text

**System Role:** You are an expert Synthetic Data Generator for training LLM agents for a Calendar Model Context Protocol (MCP).

**Objective:** Generate a synthetic training dataset containing User Queries and Expected Agent Behaviors.

**Constraint:** Only generate queries that reference days and events existing within the calender range.

**Instructions:**

1. **Generate Data:** For each of the 7 categories below, generate **2 distinct examples** that reference the specific events/dates in the calender.
2. **Categories:**
* Schedule a Single Event
* Vague & Contextual (Reasoning Required)
* Modifier & Correction (Rescheduling/Updates)
* Information Retrieval (Querying)
* Complex Logic & Conflict (Advanced)
* Human Chaos (Edge Cases/Fragments)
* Relative & Recurring (within simulation bounds)



**Required Output Format (JSON):**

```json
[
  {
    "category": str # "The category of example generated"
    "query": str # "The natural language user input",
    "complexity": str # "Low|Medium|High",
    "expected_behavior": str # "a small textual description of what the agent should do for this query",
    "addressed_days": List[str] # [list of days where the query is expected to perform],
    "current_time": YYYY-MM-DDTHH:MM:SS # The current time of querying - gives context for queries with "today", "tomorrow", "yesterday",
  }
]
''')

# The "addressed_days" will be used to preserve tokens when giving calender states to LMM as a judge by only passing 
# the before and after states of relevant days
# The "expected_behavior" may be during eval - not sure right now, so generating it anyway

# Example
'''[
  {
    "category": "Schedule a Single Event",
    "query": "I need to add a quick 30-minute touch-base with the Surgery Lead this Thursday at 3:30 PM to follow up on the board presentation.",
    "complexity": "Low",
    "expected_behavior": "The agent should identify the available slot on Thursday at 15:30 (following the Board of Directors Presentation) and create a new event for 30 minutes with 'surgery_lead@regionalhealth.org'.",
    "addressed_days": ["Thursday"]
  },
  {
    "category": "Schedule a Single Event",
    "query": "Put a 'Commute to Bistro' block on my calendar for Friday at 6:30 PM before my dinner with friends.",
    "complexity": "Low",
    "expected_behavior": "The agent should create a 30-minute event from 18:30 to 19:00 on Friday, filling the gap between 'Inbox Zero' and the 'Dinner at Bistro'.",
    "addressed_days": ["Friday"]
  },
  {
    "category": "Vague & Contextual (Reasoning Required)",
    "query": "Sarah mentioned we need more time to work on the conflict resolution stuff from Wednesday. Find an hour on Friday morning to continue that coaching.",
    "complexity": "Medium",
    "expected_behavior": "The agent must recognize 'Sarah' as 'coach.sarah@executive-growth.com' and find an open one-hour slot on Friday morning (likely before 08:30 or between 10:00 and 10:30) to schedule a follow-up coaching session.",
    "addressed_days": ["Wednesday", "Friday"]
  },
  {
    "category": "Vague & Contextual (Reasoning Required)",
    "query": "The CNO mentioned a staffing crisis during our first meeting today. Please set up a follow-up with her and the HR Director tomorrow morning to check progress.",
    "complexity": "Medium",
    "expected_behavior": "The agent identifies the 'CNO' and 'HR Director' from the Monday 10:00 AM meeting and looks for a shared opening on Tuesday morning (likely 10:00 AM or earlier) to schedule the follow-up.",
    "addressed_days": ["Monday", "Tuesday"]
  },
  {
    "category": "Modifier & Correction (Rescheduling/Updates)",
    "query": "The state inspector lead pushed back our mock audit on Thursday. Move the 9 AM start to 10 AM and shorten it to just 90 minutes.",
    "complexity": "Medium",
    "expected_behavior": "The agent should modify the 'Mock Audit' on Thursday, changing the start time from 09:00 to 10:00 and adjusting the end time to 11:30.",
    "addressed_days": ["Thursday"]
  },
  {
    "category": "Modifier & Correction (Rescheduling/Updates)",
    "query": "Actually, add the Chief of Staff to my lunch with the CFO today. We need his input on the budget reallocation.",
    "complexity": "Low",
    "expected_behavior": "The agent identifies the 'Working Lunch' on Monday and adds 'chief_of_staff@regionalhealth.org' to the attendee list.",
    "addressed_days": ["Monday"]
  },
  {
    "category": "Information Retrieval (Querying)",
    "query": "When am I meeting with the legal team this week to talk about the upcoming audit?",
    "complexity": "Low",
    "expected_behavior": "The agent should search the calendar for events including 'legal@healthcarelaw.com' and 'Audit' and return the times: Monday 15:30, Wednesday 08:30, and Friday 10:30.",
    "addressed_days": ["Monday", "Wednesday", "Friday"]
  },
  {
    "category": "Information Retrieval (Querying)",
    "query": "Who is attending the supply chain meeting on Wednesday, and what time does it start?",
    "complexity": "Low",
    "expected_behavior": "The agent identifies the 'Medical Supply Chain Strategy' event at 10:00 AM on Wednesday and lists the attendees: logistics@globalsupply.com and cfo@regionalhealth.org.",
    "addressed_days": ["Wednesday"]
  },
  {
    "category": "Complex Logic & Conflict (Advanced)",
    "query": "I have an urgent request from the Mayor's office for a 1-hour call on Tuesday afternoon. Can I fit this in without moving the Insurance Reimbursement review or the Board Prep?",
    "complexity": "High",
    "expected_behavior": "The agent evaluates Tuesday afternoon; the interval between 14:30 (end of Insurance Review) and 15:00 (start of Board Prep) is only 30 minutes, and the slot after 16:00 is filled by the Workshop. The agent should report a conflict and suggest moving the 'Community Health Initiative Workshop' or finding a slot after 17:30.",
    "addressed_days": ["Tuesday"]
  },
  {
    "category": "Complex Logic & Conflict (Advanced)",
    "query": "The CFO needs to move our Surgical Wing planning on Wednesday to Thursday. Does that clash with the Board presentation or the New Hire Keynote?",
    "complexity": "High",
    "expected_behavior": "The agent checks Wednesday's 'Capital Expenditure Planning' (14:00-15:30) and attempts to move it to Thursday. It identifies that Thursday 13:30-15:30 is the Board Presentation and 17:15 is the Keynote. It should suggest the 15:30-17:00 window on Thursday as a non-conflicting alternative.",
    "addressed_days": ["Wednesday", "Thursday"]
  },
  {
    "category": "Human Chaos (Edge Cases/Fragments)",
    "query": "Dinner Friday... make it 7:30 instead. Also, I think I forgot to include my spouse in the walk on Sunday—can you add them?",
    "complexity": "Medium",
    "expected_behavior": "The agent must update the Friday 'Dinner at Bistro' start time to 19:30 and locate the 'Evening Walk' (which is actually on Wednesday, not Sunday) and clarify if the user meant the Wednesday walk or a new Sunday event.",
    "addressed_days": ["Friday", "Wednesday", "Sunday"]
  },
  {
    "category": "Human Chaos (Edge Cases/Fragments)",
    "query": "Cancel all my 1:1s today, I'm stuck in the ER.",
    "complexity": "Medium",
    "expected_behavior": "If requested on Monday, the agent identifies '1:1 Executive Coaching Session' at 17:00 and removes it. It should also scan for other events that resemble 1:1s (like the Working Lunch) and ask for confirmation.",
    "addressed_days": ["Monday"]
  },
  {
    "category": "Relative & Recurring (within simulation bounds)",
    "query": "Every day this week, I want to block out 30 minutes for 'Email Review' right before my first work meeting.",
    "complexity": "High",
    "expected_behavior": "The agent must find the first non-personal event for each day (e.g., 08:00 Mon, 08:30 Tue, 08:30 Wed, 09:00 Thu, 08:30 Fri) and create 30-minute blocks immediately preceding those times.",
    "addressed_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
  },
  {
    "category": "Relative & Recurring (within simulation bounds)",
    "query": "I need to do my 'Weekly Reflection' at the same time I have it scheduled for Sunday, but also on Saturday morning for this week only.",
    "complexity": "Medium",
    "expected_behavior": "The agent identifies the 'Rest & Reflection' block (14:00-16:00) on Sunday and creates a duplicate event on Saturday from 14:00-16:00 (or interprets 'morning' and asks for a specific Saturday time).",
    "addressed_days": ["Saturday", "Sunday"]
  }
]'''
