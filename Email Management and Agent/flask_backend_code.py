# File: app.py
# Main Flask Application for IITK Webmail Email Management System

from typing import Annotated, TypedDict
import operator
from flask import Flask, request, jsonify
from flask_cors import CORS
import imaplib
import email
from email.header import decode_header
import os
from datetime import timedelta, timezone as dt_timezone
from dateutil import parser  # Added to robustly parse ISO dates and datetimes
import json
from threading import Thread
import time
import hashlib

# AI imports
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Vector DB
import chromadb
from chromadb.config import Settings
from sklearn.neighbors import NearestNeighbors
import numpy as np
import faiss

app = Flask(__name__)
CORS(app)

# Configuration
IMAP_SERVER = "newmailhost.cc.iitk.ac.in"
IMAP_PORT = 993
GEMINI_API_KEY = "" #Enter your Gemini Api key here

# OAuth dev port and redirect URI - keep this in sync with Google Cloud Console
OAUTH_PORT = 8080
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_PORT}/"

# Global Mails Array
MAILS_ARRAY = []

# Initialize FAISS index
embedding_dim = 3072  # Adjust based on your embeddings model
faiss_index = faiss.IndexFlatL2(embedding_dim)
email_metadata = {}  # Dictionary to store metadata for each email

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyAwQhQ_FVR2ZmheVLbVxdUi9JUWOVF5rfM",
    temperature=0
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key="AIzaSyDDHaQDmegZM01578UWCSC8CN749L9Sbig"
)

# Wrap the embeddings model in a callable function
def embedding_function(text):
    """Generate embeddings for the given text."""
    try:
        embeddings = embeddings_model.embed_query(text)
        # print(f"Generated embeddings: {embeddings}")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise

# Add email to FAISS index
def add_email_to_faiss(email_id, email_text, metadata):
    try:
        embedding = embedding_function(email_text)
        faiss_index.add(np.array([embedding], dtype=np.float32))
        print("Email added to FAISS index")
        # print size of index
        print(f"FAISS index size: {faiss_index.ntotal}")
        email_metadata[email_id] = metadata
    except Exception as e:
        print(f"Error adding email to FAISS: {e}")

# Retrieve emails from FAISS index
def retrieve_emails_from_faiss(query, k=1):
    try:
        query_embedding = embedding_function(query)
        distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k)
        results = []
        for idx in indices[0]:
            if idx != -1:  # Ensure valid index
                email_id = list(email_metadata.keys())[idx]
                results.append(email_metadata[email_id])
        print(f"length of FAISS index: {faiss_index.ntotal}")
        print(f"Retrieved {len(results)} emails from FAISS")
        return results
    except Exception as e:
        print(f"Error retrieving emails from FAISS: {e}")
        return []

# ==================== EMAIL FETCHING ====================

class EmailFetcher:
    def __init__(self, email_address, password, classifier):
        self.email_address = email_address
        self.password = password
        self.classifier = classifier
        self.imap = None

    def connect(self):
        """Connect to IMAP server"""
        try:
            self.imap = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
            self.imap.login(self.email_address, self.password)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def fetch_emails(self, num_emails=1):
        """Fetch top N emails and store unique ones in local storage and FAISS index"""
        if not self.imap:
            if not self.connect():
                return []

        try:
            self.imap.select('INBOX')
            _, message_numbers = self.imap.search(None, 'ALL')

            email_ids = message_numbers[0].split()
            email_ids = email_ids[-num_emails:]  # Get last N emails
            email_ids.reverse()  # Most recent first

            # Load existing email hashes from storage
            if os.path.exists('storage.txt'):
                with open('storage.txt', 'r') as f:
                    stored_data = json.load(f)
            else:
                stored_data = {"emails": [], "hashes": []}

            existing_hashes = set(stored_data.get("hashes", []))
            new_emails = []

            for email_id in email_ids:
                _, msg_data = self.imap.fetch(email_id, '(RFC822)')

                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        # Parse email
                        subject = self.decode_subject(msg.get('Subject', ''))
                        from_addr = msg.get('From', '')
                        date = msg.get('Date', '')

                        # Get email body
                        body = self.get_email_body(msg)

                        # Create a unique hash for the email
                        email_content = f"{subject}{from_addr}{date}{body}"
                        email_hash = hashlib.sha256(email_content.encode('utf-8')).hexdigest()

                        if email_hash not in existing_hashes:
                            email_obj = {
                                'id': email_id.decode(),
                                'subject': subject,
                                'from': from_addr,
                                'date': date,
                                'body': body,
                                'hash': email_hash
                            }

                            # Classify the new email
                            classification = self.classifier.classify_email(email_obj)
                            email_obj['classification'] = classification

                            new_emails.append(email_obj)

                            # Prepend the new email so newest emails are at the top
                            stored_data["emails"] = [email_obj] + stored_data.get("emails", [])
                            existing_hashes.add(email_hash)

                            # Keep hashes in sync; prepend newest hash
                            stored_data["hashes"] = [email_hash] + stored_data.get("hashes", [])

                            # Add to FAISS index
                            email_text = f"Subject: {subject}\nFrom: {from_addr}\nBody: {body}"
                            metadata = {
                                'id': email_id.decode(),
                                'subject': subject,
                                'from': from_addr,
                                'date': date,
                                'body': body,
                                'classification': classification
                            }
                            add_email_to_faiss(email_id.decode(), email_text, metadata)

            # Save updated storage data
            with open('storage.txt', 'w') as f:
                json.dump(stored_data, f, indent=4)

            # Save FAISS index and metadata
            save_faiss_index()

            return new_emails

        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []

    def decode_subject(self, subject):
        """Decode email subject"""
        if subject:
            decoded = decode_header(subject)
            subject = ''
            for content, encoding in decoded:
                if isinstance(content, bytes):
                    subject += content.decode(encoding or 'utf-8', errors='ignore')
                else:
                    subject += content
        return subject

    def get_email_body(self, msg):
        """Extract email body"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = msg.get_payload()

        return body

    def close(self):
        if self.imap:
            self.imap.close()
            self.imap.logout()



# ==================== EMAIL CLASSIFIER ====================

class EmailClassifier:
    def __init__(self):
        self.llm = llm

    def classify_email(self, email_obj):
        """Classify email into categories using Gemini with structured output"""
        
        prompt = f"""
You are an email classifier for IITK students. Classify the following email into ONE of these categories:
1. academic - Course-related emails, assignments, quizzes, exams, grades
2. club - Club activities, events, meetings, announcements
3. general - Administrative, hostel, placements, other announcements

Email Details:
From: {email_obj['from']}
Subject: {email_obj['subject']}
Body: {email_obj['body'][:1000]}  # First 1000 chars

Return the JSON object ONLY:
{{"category": "academic|club|general"}}

Do not include any markdown formatting, quotes, or additional text.
"""

        try:
            response = self.llm.invoke(prompt)
            # Clean the response by removing markdown code block and whitespace
            clean_response = response.content.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('\n')[1:-1][0]
            print(clean_response)

            result = json.loads(clean_response)
            return result
        except Exception as e:
            print(f"Classification error: {e}")
            return {"category": "general"}

    def classify_batch(self, emails):
        """Classify multiple emails"""
        for email_obj in emails:
            classification = self.classify_email(email_obj)
            email_obj['classification'] = classification
        return emails



from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import timedelta, timezone as dt_timezone
from dateutil import parser  # Added to robustly parse ISO dates and datetimes
import os.path
import pickle
import json

# ============== Tools ===================
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    """Authenticate and return Google Calendar service object"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            # Use a static port for development and testing. Make sure this exact
            # redirect URI (including trailing slash) is added to Google Cloud Console
            # under 'Authorized redirect URIs' for your OAuth client.
            creds = flow.run_local_server(port=OAUTH_PORT, host='localhost')
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('calendar', 'v3', credentials=creds)


def add_event(summary, start_time, end_time, description="", location="", 
              attendees=None, timezone="UTC"):
    """
    Add an event to Google Calendar with optional recurrence
    
    Args:
        summary (str): Event title/summary
        start_time (str): Start time in ISO format (e.g., "2025-11-15T10:00:00")
        end_time (str): End time in ISO format (e.g., "2025-11-15T11:00:00")
        description (str): Event description (optional)
        location (str): Event location (optional)
        attendees (list): List of email addresses (optional)
        timezone (str): Timezone (default: "UTC")
    
    Returns:
        dict: Created event details or error message
    """
    try:
        service = get_calendar_service()

        # Check for clashes before creating the event
        clash_result = check_clash(start_time, end_time, timezone=timezone)
        if clash_result.get('has_clash'):
            return {
                'success': False,
                'error': 'Scheduling conflict',
                'details': clash_result
            }

        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': timezone,
            },
        }

        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]

        event = service.events().insert(calendarId='primary', body=event).execute()

        return {
            'success': True,
            'event_id': event['id'],
            'summary': event['summary'],
            'start': event['start']['dateTime'],
            'end': event['end']['dateTime'],
            'link': event.get('htmlLink')
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# ==================================================================
# START: This is the new, more robust version
# ==================================================================

# Import the built-in datetime classes and the dateutil modules
from datetime import datetime, timezone as dt_timezone, timedelta
from dateutil import parser, tz

# --- DEFINE IST MANUALLY ---
# Create an IST timezone object (+5 hours, 30 minutes)
IST = dt_timezone(timedelta(hours=5, minutes=30))

def check_clash(start_time, end_time, timezone="UTC", exclude_event_id=None):
    """
    Check for clashes, intelligently defaulting to IST.
    
    If the provided timezone is "UTC" (the agent's default) or None,
    this function will assume IST.
    If a specific timezone (e.g., "Europe/London") is provided, it will be used.
    """
    try:
        service = get_calendar_service()

        # --- 1. Get Timezone Object (With IST Default Logic) ---

        # <-- LOGIC CHANGED HERE
        if timezone is None or timezone.upper() == "UTC":
            # If agent provides no timezone OR the generic "UTC" default,
            # override it with our hard-coded IST.
            local_tz = IST
        else:
            # The agent provided a specific timezone (e.g., "Europe/London").
            # Let's try to use it.
            local_tz = tz.gettz(timezone)
            if local_tz is None:
                # If the string was invalid, fall back to IST.
                local_tz = IST
        # <-- END OF CHANGED LOGIC

        # --- 2. Parse and Standardize New Event Times ---
        new_start_dt = parser.isoparse(start_time)
        new_end_dt = parser.isoparse(end_time)

        # If the parsed time is naive, apply the 'local_tz' we just determined.
        if new_start_dt.tzinfo is None:
            new_start_dt = new_start_dt.replace(tzinfo=local_tz)
        if new_end_dt.tzinfo is None:
            new_end_dt = new_end_dt.replace(tzinfo=local_tz)

        # --- 3. Query the API ---
        time_min_utc = new_start_dt.astimezone(dt_timezone.utc).isoformat()
        time_max_utc = new_end_dt.astimezone(dt_timezone.utc).isoformat()

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min_utc,
            timeMax=time_max_utc,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        clashing_events = []

        # --- 4. Loop and Compare ---
        for event in events:
            if exclude_event_id and event['id'] == exclude_event_id:
                continue

            event_start_str = event['start'].get('dateTime', event['start'].get('date'))
            event_end_str = event['end'].get('dateTime', event['end'].get('date'))

            evt_start_dt = parser.isoparse(event_start_str)
            evt_end_dt = parser.isoparse(event_end_str)

            if evt_start_dt.tzinfo is None:
                evt_start_dt = evt_start_dt.replace(tzinfo=dt_timezone.utc)
            if evt_end_dt.tzinfo is None:
                evt_end_dt = evt_end_dt.replace(tzinfo=dt_timezone.utc)
            
            if (new_start_dt < evt_end_dt and new_end_dt > evt_start_dt):
                clashing_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'start': event_start_str,
                    'end': event_end_str,
                    'link': event.get('htmlLink')
                })

        has_clash = len(clashing_events) > 0

        return {
            'has_clash': has_clash,
            'clash_count': len(clashing_events),
            'clashing_events': clashing_events,
            'message': f"Found {len(clashing_events)} conflicting event(s)" if has_clash else "No conflicts found"
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# ==================================================================
# END: Replacement section
# ==================================================================

# # Import the built-in datetime classes
# from datetime import datetime, timezone as dt_timezone, timedelta
# from dateutil import parser

# # --- DEFINE IST MANUALLY ---
# # Create an IST timezone object (+5 hours, 30 minutes)
# IST = dt_timezone(timedelta(hours=5, minutes=30)) # <-- NEW LINE

# # Assuming get_calendar_service() is defined elsewhere
# # from your_auth_file import get_calendar_service 

# def check_clash(start_time, end_time, timezone="UTC", exclude_event_id=None):
#     """
#     Check if there's a clash with existing events at given date and time
    
#     Args:
#         start_time (str): Start time in ISO format (e.g., "2025-11-15T10:00:00")
#         end_time (str): End time in ISO format (e.g., "2025-11-15T11:00:00")
#         timezone (str): Timezone (default: "UTC") - NOTE: This param is unused
#         exclude_event_id (str): Event ID to exclude from clash check (optional)
    
#     Returns:
#         dict: Clash status and conflicting events if any
#     """
#     try:
#         service = get_calendar_service()

#         # --- 1. Parse and Standardize New Event Times ---
#         new_start_dt = parser.isoparse(start_time)
#         new_end_dt = parser.isoparse(end_time)

#         # If the parsed time is naive (no timezone), assume IST.
#         if new_start_dt.tzinfo is None:
#             new_start_dt = new_start_dt.replace(tzinfo=IST) # <-- CHANGED
#         if new_end_dt.tzinfo is None:
#             new_end_dt = new_end_dt.replace(tzinfo=IST) # <-- CHANGED

#         # --- 2. Query the API ---
#         # Now convert the IST-aware time to UTC for the API call
#         time_min_utc = new_start_dt.astimezone(dt_timezone.utc).isoformat()
#         time_max_utc = new_end_dt.astimezone(dt_timezone.utc).isoformat()

#         events_result = service.events().list(
#             calendarId='primary',
#             timeMin=time_min_utc,
#             timeMax=time_max_utc,
#             singleEvents=True,
#             orderBy='startTime'
#         ).execute()

#         events = events_result.get('items', [])
#         clashing_events = []

#         # --- 3. Loop and Compare ---
#         for event in events:
#             if exclude_event_id and event['id'] == exclude_event_id:
#                 continue

#             event_start_str = event['start'].get('dateTime', event['start'].get('date'))
#             event_end_str = event['end'].get('dateTime', event['end'].get('date'))

#             evt_start_dt = parser.isoparse(event_start_str)
#             evt_end_dt = parser.isoparse(event_end_str)

#             # If it was an all-day 'date' string, make it aware
#             # We use UTC here, as all-day events are timezone-neutral
#             if evt_start_dt.tzinfo is None:
#                 evt_start_dt = evt_start_dt.replace(tzinfo=dt_timezone.utc) # <-- Reverted to UTC (Correct for all-day)
#             if evt_end_dt.tzinfo is None:
#                 evt_end_dt = evt_end_dt.replace(tzinfo=dt_timezone.utc) # <-- Reverted to UTC (Correct for all-day)
                
#             # Check for overlap
#             if (new_start_dt < evt_end_dt and new_end_dt > evt_start_dt):
#                 clashing_events.append({
#                     'id': event['id'],
#                     'summary': event.get('summary', 'No title'),
#                     'start': event_start_str,
#                     'end': event_end_str,
#                     'link': event.get('htmlLink')
#                 })

#         has_clash = len(clashing_events) > 0

#         return {
#             'has_clash': has_clash,
#             'clash_count': len(clashing_events),
#             'clashing_events': clashing_events,
#             'message': f"Found {len(clashing_events)} conflicting event(s)" if has_clash else "No conflicts found"
#         }

#     except Exception as e:
#         return {'success': False, 'error': str(e)}


# def check_clash(start_time, end_time, timezone="UTC", exclude_event_id=None):
#     """
#     Check if there's a clash with existing events at given date and time
    
#     Args:
#         start_time (str): Start time in ISO format (e.g., "2025-11-15T10:00:00")
#         end_time (str): End time in ISO format (e.g., "2025-11-15T11:00:00")
#         timezone (str): Timezone (default: "UTC")
#         exclude_event_id (str): Event ID to exclude from clash check (optional)
    
#     Returns:
#         dict: Clash status and conflicting events if any
#     """
#     try:
#         service = get_calendar_service()

#         # --- 1. Parse and Standardize New Event Times ---
#         # Use dateutil.parser.isoparse() as it handles both 'date' and 'dateTime'
#         new_start_dt = parser.isoparse(start_time)
#         new_end_dt = parser.isoparse(end_time)

#         # If the parsed time is naive (no timezone), assume UTC.
#         if new_start_dt.tzinfo is None:
#             new_start_dt = new_start_dt.replace(tzinfo=dt_timezone.utc)
#         if new_end_dt.tzinfo is None:
#             new_end_dt = new_end_dt.replace(tzinfo=dt_timezone.utc)

#         # --- 2. Query the API ---
#         time_min_utc = new_start_dt.isoformat()
#         time_max_utc = new_end_dt.isoformat()

#         events_result = service.events().list(
#             calendarId='primary',
#             timeMin=time_min_utc,
#             timeMax=time_max_utc,
#             singleEvents=True,
#             orderBy='startTime'
#         ).execute()

#         events = events_result.get('items', [])
#         clashing_events = []

#         # --- 3. Loop and Compare ---
#         for event in events:
#             # Skip if this is the event we're updating
#             if exclude_event_id and event['id'] == exclude_event_id:
#                 continue

#             # Get the raw start/end time string, whether it's 'date' or 'dateTime'
#             event_start_str = event['start'].get('dateTime', event['start'].get('date'))
#             event_end_str = event['end'].get('dateTime', event['end'].get('date'))

#             # Parse it. parser.isoparse() handles both
#             evt_start_dt = parser.isoparse(event_start_str)
#             evt_end_dt = parser.isoparse(event_end_str)

#             # If it was an all-day 'date' string, it's now naive; make it UTC-aware
#             if evt_start_dt.tzinfo is None:
#                 evt_start_dt = evt_start_dt.replace(tzinfo=dt_timezone.utc)
#             if evt_end_dt.tzinfo is None:
#                 evt_end_dt = evt_end_dt.replace(tzinfo=dt_timezone.utc)

#             # Check for overlap
#             if (new_start_dt < evt_end_dt and new_end_dt > evt_start_dt):
#                 clashing_events.append({
#                     'id': event['id'],
#                     'summary': event.get('summary', 'No title'),
#                     'start': event_start_str,
#                     'end': event_end_str,
#                     'link': event.get('htmlLink')
#                 })

#         has_clash = len(clashing_events) > 0

#         return {
#             'has_clash': has_clash,
#             'clash_count': len(clashing_events),
#             'clashing_events': clashing_events,
#             'message': f"Found {len(clashing_events)} conflicting event(s)" if has_clash else "No conflicts found"
#         }

#     except Exception as e:
#         return {'success': False, 'error': str(e)}

# ===================== Agents =====================
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


def retriever_function(query: str, k: int = 1):
    """Custom retriever function for FAISS"""


    print(f"Retrieving top {k} documents for query: {query}")
    try:
        results = retrieve_emails_from_faiss(query, k)
        # Format results to match expected structure
        documents = []
        for result in results:
            doc_content = f"Subject: {result['subject']}\nFrom: {result['from']}\nBody: {result['body'][:2000]}"  # First 2000 chars
            documents.append(doc_content)
        return documents
    except Exception as e:
        print(f"Retriever error: {e}")
        return []

rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for answering questions about the user's emails. "
                   "Use the provided context to answer the question. If you don't know the answer, say so.\n"
                   "Context:\n{context}"),
        ("human", "Question:\n{question}")
    ])

def format_docs(docs):
    """Format retrieved documents into a single string"""
    context = "\n\n".join([d for d in docs])
    print(f"Debugging Context:\n{context}")  # Add this line to print the context
    return context

rag_chain = (
    {"context": lambda x: format_docs(retriever_function(x["question"])),
     "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

@tool
def email_rag_tool(question: str) -> str:
    """Tool to answer questions about emails using RAG"""
    return rag_chain.invoke({"question": question})

class CheckClashArgs(BaseModel):
    start_time: str = Field(..., description="Start time in ISO format (e.g., '2025-11-15T10:00:00')")
    end_time: str = Field(..., description="End time in ISO format (e.g., '2025-11-15T11:00:00')")
    timezone: Optional[str] = Field("UTC", description="Timezone (default: 'UTC')")
    exclude_event_id: Optional[str] = Field(None, description="Event ID to exclude from clash check")

@tool(args_schema=CheckClashArgs)
def check_clash_tool(start_time: str, end_time: str, timezone: Optional[str] = "UTC", exclude_event_id: Optional[str] = None) -> str:
    """Tool to check for calendar event clashes (accepts named args)."""
    result = check_clash(
        start_time=start_time,
        end_time=end_time,
        timezone=timezone,
        exclude_event_id=exclude_event_id
    )
    return json.dumps(result)

class AddEventArgs(BaseModel):
    summary: str = Field(..., description="Event title/summary")
    start_time: str = Field(..., description="Start time in ISO format (e.g., '2025-11-15T10:00:00')")
    end_time: str = Field(..., description="End time in ISO format (e.g., '2025-11-15T11:00:00')")
    description: Optional[str] = Field("", description="Event description")
    location: Optional[str] = Field("", description="Event location")
    attendees: Optional[List[str]] = Field(None, description="List of attendee email addresses")
    timezone: Optional[str] = Field("UTC", description="Timezone (default: 'UTC')")

@tool(args_schema=AddEventArgs)
def add_event_tool(
    summary: str, 
    start_time: str, 
    end_time: str, 
    description: Optional[str] = "", 
    location: Optional[str] = "", 
    attendees: Optional[List[str]] = None, 
    timezone: Optional[str] = "UTC"
) -> str:
    """Tool to add an event to Google Calendar"""
    # Note: I also fixed the argument order to match the add_event definition
    result = add_event(
        summary=summary,
        start_time=start_time,
        end_time=end_time,
        description=description,
        location=location,
        attendees=attendees,
        timezone=timezone
    )
    return json.dumps(result)

tools = [email_rag_tool, check_clash_tool, add_event_tool]
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

# Define the agent's system prompt
SYSTEM_PROMPT = """You are an AI assistant that helps users manage their emails and calendar.
For ANY questions about emails, email content, or searching through emails, you MUST use the email_rag_tool.
Only use the calendar tools (check_clash_tool, add_event_tool) for calendar-related operations.
Never try to answer email-related questions directly without using the email_rag_tool."""

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define the primary node for the agent
def call_model(state: AgentState):
    """The primary node for the agent. Calls the LLM with the current messages."""
    print("--- 🧠 Calling Gemini ---")
    messages = state["messages"]
    
    # Add system message to guide the model
    system_message = """You are an AI assistant that helps users manage their emails and calendar.
For ANY questions about emails, email content, or searching through emails, you MUST use the email_rag_tool.
Only use the calendar tools (check_clash_tool, add_event_tool) for calendar-related operations.
Never try to answer email-related questions directly without using the email_rag_tool."""
    
    messages = [HumanMessage(content=system_message)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



def should_continue(state: AgentState) -> str:
    print("--- Routing---")
    last_message = state["messages"][-1]

    # If the LLM proposed tool calls, run them
    if getattr(last_message, "tool_calls", None):
        print("-> Decision: Call tool(s)")
        return "call_tool"

    # If no tool calls from LLM, we can end and return its answer
    print("-> Decision: End (respond to user)")
    return END


# Assemble the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        END: END
    }
)
workflow.add_edge("call_tool", "agent")

# Compile the graph
app1 = workflow.compile()

def run_query(query: str):
    print(f"\n=========================================")
    print(f"🚀 USER QUERY: {query}")
    print(f"=========================================\n")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    try:
        for event in app1.stream(inputs, {"recursion_limit": 10}):
            if "agent" in event:
                print(event["agent"]["messages"][-1])
            if "call_tool" in event:
                print(event["call_tool"]["messages"][-1])
            print("---")
        print("\n✅ Query finished.")
        return {"response": event["agent"]["messages"][-1].content}
    except Exception as e:
        print(f"🚨🚨 AN ERROR OCCURRED: {e} 🚨🚨")
        print("This is likely due to an invalid API key or network issue.")

# ==================== FLASK ROUTES ====================

# Global variables for email credentials (in production, use secure storage)
# Email credentials are stored separately from Google API credentials to reduce blast radius.
EMAIL_CREDENTIALS = {
    'email': None,
    'password': None
}

# Try to load email credentials from `email_credentials.json`. This file should contain a JSON
# object with `email` and `password` fields. Do NOT commit real credentials to source control.
try:
    if os.path.exists('email_credentials.json'):
        with open('email_credentials.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            EMAIL_CREDENTIALS['email'] = data.get('email')
            EMAIL_CREDENTIALS['password'] = data.get('password')
except Exception as e:
    print(f"Warning: could not load email_credentials.json: {e}")

#serve index.html at home route
@app.route('/', methods=['GET'])
def home():
    """Serve the index.html file"""
    fetch_emails_endpoint()
    return app.send_static_file('index.html')


@app.route('/login', methods=['POST'])
def login():
    """Store email credentials (for demo purposes)"""
    data = request.json
    email_addr = data.get('email')
    password = data.get('password')

    EMAIL_CREDENTIALS['email'] = email_addr
    EMAIL_CREDENTIALS['password'] = password

    # Persist to email_credentials.json so the server can restart without losing login
    try:
        with open('email_credentials.json', 'w', encoding='utf-8') as f:
            json.dump({'email': email_addr, 'password': password}, f, indent=2)
    except Exception as e:
        print(f"Error saving email credentials: {e}")

    return jsonify({"status": "success", "message": "Credentials stored"})

@app.route('/fetch_emails', methods=['GET'])
def fetch_emails_endpoint():
    """Fetch and classify emails"""

    global MAILS_ARRAY

    if not EMAIL_CREDENTIALS:
        return jsonify({"error": "Not logged in"}), 401

    try:
        # Fetch emails
        classifier = EmailClassifier()
        fetcher = EmailFetcher(
            EMAIL_CREDENTIALS['email'],
            EMAIL_CREDENTIALS['password'],
            classifier
        )
        emails = fetcher.fetch_emails(num_emails=10)
        fetcher.close()

        # Store in global array
        MAILS_ARRAY = emails

        

        return jsonify({
            "status": "success",
            "count": len(emails),
            "emails": MAILS_ARRAY
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_emails', methods=['GET'])
def get_emails():
    """Get stored emails"""

    global MAILS_ARRAY

    # get all emails from storage.txt
    if os.path.exists('storage.txt'):
        with open('storage.txt', 'r') as f:
            stored_data = json.load(f)
            MAILS_ARRAY = stored_data.get("emails", [])

    return jsonify({
        "status": "success",
        "count": len(MAILS_ARRAY),
        "emails": MAILS_ARRAY
    })


@app.route('/run_query', methods=['GET'])
def run_query_endpoint():
    """Run a query against the email data"""
    query = request.args.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    response = run_query(query)
    return jsonify(response)



# ==================== BACKGROUND TASK ====================

def background_email_fetcher():
    """Background task to periodically fetch emails"""
    while True:
        if EMAIL_CREDENTIALS:
            try:
                print("Fetching emails in background...")
                # Call fetch endpoint logic
                time.sleep(3000)  # Fetch every 50 minutes
            except Exception as e:
                print(f"Background fetch error: {e}")
        time.sleep(60)

# Add persistence methods for FAISS index
import pickle

# Save FAISS index and metadata to local files
def save_faiss_index(index_path="faiss_index", metadata_path="email_metadata.pkl"):
    try:
        faiss.write_index(faiss_index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(email_metadata, f)
        print("FAISS index and metadata saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

# Load FAISS index and metadata from local files
def load_faiss_index(index_path="faiss_index", metadata_path="email_metadata.pkl"):
    global faiss_index, email_metadata
    try:
        faiss_index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            email_metadata = pickle.load(f)
        print("FAISS index and metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")

# Ensure FAISS index is loaded at startup
if os.path.exists("faiss_index") and os.path.exists("email_metadata.pkl"):
    load_faiss_index()
else:
    print("No saved FAISS index found. Starting with an empty index.")

if __name__ == '__main__':
    # Start background thread (optional)
    # background_thread = Thread(target=background_email_fetcher, daemon=True)
    # background_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000)
