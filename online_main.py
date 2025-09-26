# main.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Annotated, Optional  # <-- Import Optional
import operator
import asyncio
from client import EmailMCPClient
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import json
# 1) Load .env that sits NEXT TO THIS FILE (robust even if launched elsewhere)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# 2) Read + sanity-check the key
KEY = os.getenv("GEMINI_API_KEY", "")
if not KEY or KEY.strip() != KEY or len(KEY) < 20:
    raise RuntimeError(
        "GEMINI_API_KEY is missing or has leading/trailing spaces. "
        "Put it in a .env next to main.py as: GEMINI_API_KEY=AI... (no quotes)."
    )
# Some parts of the wrapper look for GOOGLE_API_KEY; set it too.
os.environ.setdefault("GOOGLE_API_KEY", KEY)

# 3) Prove the key works with the RAW SDK BEFORE using the wrapper (one-time check)
import google.generativeai as genai
genai.configure(api_key=KEY)
try:
    _probe = genai.GenerativeModel("gemini-2.5-flash").generate_content("ping")
    _ = getattr(_probe, "text", "")  # don't print, just ensure we got a response
except Exception as e:
    raise RuntimeError(f"Gemini API key rejected by SDK: {e}")

# 4) Now build the LangChain wrapper with the SAME key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=KEY,   # explicit -> avoids ADC / env ambiguity
)
import re
from datetime import datetime, timedelta

def _parse_query_fast(user_text: str) -> dict:
    t = (user_text or "").strip()
    d = {"sender_like":"","subject_like":"","priority_in":"",
         "window":"","date_from":"","date_to":"",
         "semantic_query":"","topic_hint":""}
    if not t: 
        return d

    # sender/domain
    m = re.search(r"from:\s*([^\s,]+)", t, re.I)
    if m: d["sender_like"] = m.group(1)

    # subject:
    m = re.search(r"subject:\s*([^,]+)", t, re.I)
    if m: d["subject_like"] = m.group(1).strip()

    # priority
    if re.search(r"\burgent|high priority\b", t, re.I): d["priority_in"] = "High"
    elif re.search(r"\blow priority\b", t, re.I): d["priority_in"] = "Low"

    # windows
    # windows (order matters)
    if re.search(r"\blast month\b", t, re.I):
        # previous calendar month window
        today = datetime.now()
        first_this = datetime(today.year, today.month, 1)
        last_month_end = first_this - timedelta(days=1)
        last_month_start = datetime(last_month_end.year, last_month_end.month, 1)
        d["date_from"] = last_month_start.strftime("%Y-%m-%d")
        d["date_to"]   = last_month_end.strftime("%Y-%m-%d")
    elif re.search(r"\btoday\b", t, re.I):
        d["window"] = "today"
    elif re.search(r"\byesterday\b", t, re.I):
        d["window"] = "yesterday"
    # [REPLACE lines 79–83]
    elif re.search(r"\blast\s+7d|\blast week\b", t, re.I):
        d["window"] = "last 7 days"   # server understands this
    elif re.search(r"\blast\s+30d\b|\bpast 30 days\b", t, re.I):
        # server _win() doesn’t have 30d by default; set explicit range
        today = datetime.now()
        d["date_from"] = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        d["date_to"]   = today.strftime("%Y-%m-%d")

    else:
        m = re.search(r"(\d{4})-(\d{2})", t)  # YYYY-MM
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            d["date_from"] = f"{y}-{mo:02d}-01"
            # naive month-end
            next_mo = datetime(y, mo, 1) + timedelta(days=32)
            end = datetime(next_mo.year, next_mo.month, 1) - timedelta(days=1)
            d["date_to"] = end.strftime("%Y-%m-%d")

    # semantic free text
    if not any(d[k] for k in ["sender_like","subject_like","priority_in","window","date_from","date_to"]):
        d["semantic_query"] = t
        d["topic_hint"] = t[:64]
    return d

EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)

def _extract_account(text: str) -> str:
    t = (text or "").lower()
    m = EMAIL_RE.search(t)
    if m:
        return m.group(0)
    if "secondary" in t:
        return "secondary"
    if "primary" in t:
        return "primary"
    return "primary"
from typing import Optional

def _maybe_extract_explicit_account(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = EMAIL_RE.search(t)
    if m:
        return m.group(0)       # explicit email
    if "secondary" in t:
        return "secondary"      # explicit alias
    if "primary" in t:
        return "primary"        # explicit alias
    return None

# [MODIFIED] - Add draft_id to the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_task: str
    email_context: dict
    user_intent: str
    draft_id: Optional[str]  # <-- To hold the ID of the draft between turns
    active_account: Optional[str] # <-- To hold the resolved account email

class GmailAgent:
    def __init__(self, mcp_client, llm):
        self.mcp_client = mcp_client
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify", self.classify_intent)
        graph.add_node("search", self.search_emails)
        graph.add_node("summarize", self.summarize_emails)
        graph.add_node("draft", self.draft_emails)
        graph.add_node("send", self.send_email)  # <-- Add new 'send' node
        graph.add_node("respond", self.generate_response)

        # [MODIFIED] - All routing is now conditional, starting from 'classify'
        graph.add_conditional_edges(
            "classify",
            self.route_intent,
            {
                "SEARCH": "search",
                "SUMMARIZE": "search",  # Summarize needs to search first
                "DRAFT": "draft",
                "SEND": "send",
                "OTHER": "respond"
            }
        )
        graph.add_conditional_edges(
            "search", self.route_after_search,
        )
        graph.add_edge("summarize", END)
        graph.add_edge("draft", END)  # The draft node will now end the turn to wait for user confirmation
        graph.add_edge("send", END)  # The send node is also a final step
        graph.add_edge("respond", END)
        graph.set_entry_point("classify")
        return graph

    # [MODIFIED] - Updated classifier to include SEND intent
    async def classify_intent(self, state: AgentState):
        user_message = state["messages"][-1].content

        # Simple rule-based check for confirmation, which is more reliable than an LLM call for "yes/no"
        if state.get("draft_id") and user_message.lower().strip() in ["yes", "y", "ok", "send it", "please send"]:
            return {**state, "user_intent": "SEND"}
        # Fast rule-based check first
        txt = user_message.lower()
        if any(k in txt for k in ["draft", "compose", "reply", "respond"]):
            intent = "DRAFT"
        elif any(k in txt for k in ["summarize", "tl;dr", "brief"]):
            intent = "SUMMARIZE"
        elif any(k in txt for k in ["find", "search", "show", "look for"]):
            intent = "SEARCH"
        else:
            # Fallback to LLM only if still unclear
            classification_prompt = (
                "You are an expert intent classifier. What is the user's intent in this message: "
                "SEARCH, SUMMARIZE, DRAFT or OTHER?\n"
                f"User message: {user_message}\n"
                "Respond with only one of: SEARCH, SUMMARIZE, DRAFT, OTHER."
            )
            response = await self.llm.ainvoke(classification_prompt)
            intent = response.content.strip().upper()
        return {**state, "user_intent": intent, "current_task": "classified"}
        
    # [NEW] - Router function for the classifier
    async def route_intent(self, state: AgentState):
        return state["user_intent"]
    
    # [NEW] - Router function for the classifier
    async def search_emails(self, state: AgentState):
        user_message = state["messages"][-1].content

        args = _parse_query_fast(user_message)
        # New search request: clear any previously selected email so summarize doesn't reuse it
        if state.get("user_intent") == "SEARCH":
            state["selected_email_id"] = None

        # only if clearly ambiguous AND user intent is SEARCH, ask LLM once
        if (state["user_intent"] == "SEARCH" and not any(args[k] for k in
            ["sender_like","subject_like","priority_in","window","date_from","date_to","semantic_query"])):
            arg_prompt = (
                "Extract filters from this message as strict JSON with keys: "
                "sender_like, subject_like, priority_in, window, date_from, date_to, semantic_query, topic_hint. "
                "Use empty strings when unknown.\n"
                f"Message: {user_message}\nJSON:"
            )
            try:
                arg_resp = await self.llm.ainvoke(arg_prompt)
                args = json.loads(arg_resp.content)
            except Exception:
                pass

        # Always include account for the server.
        # Default to session's active account; switch only if the user explicitly asked.
        user_message = state["messages"][-1].content
        account = state.get("active_account") or "primary"
        explicit = _maybe_extract_explicit_account(user_message)
        if explicit is not None:
            account = explicit  # switch for this and future turns
        args["account_email"] = account

        # 2) Call local RAG tool first
        def _desired_limit(user_text: str, intent: str) -> int:
            t = (user_text or "").lower()
            # explicit "top N"
            import re
            m = re.search(r"\btop\s+(\d{1,3})\b", t)
            if m:
                return max(1, min(50, int(m.group(1))))
            # plural cues
            if any(w in t for w in ["all mails", "all emails", "summaries", "list", "many"]):
                return 20  # sane upper bound for interactive UX
            # single summary phrasing
            if intent == "SUMMARIZE" and any(w in t for w in ["this mail", "that mail", "the mail"]):
                return 1
            # default
            return 5

        lim = _desired_limit(user_message, state.get("user_intent",""))


        # If we have a previously selected email, skip search
        sel = state.get("selected_email_id")
        if sel and state.get("user_intent") == "SUMMARIZE":
            return {
                **state,
                "email_context": {"search_query": args, "search_results": [{"email_id": sel}]},
                "current_task": "searched", "active_account": account,
            }

        rag_json = await self.mcp_client.call_tool("rag_search_emails", {**args, "limit": lim})

        local_items = []
        try:
            parsed = json.loads(rag_json or "{}")
            status = parsed.get("status")
            if status == "one":
                local_items = [parsed.get("item") or {}]
            elif status == "ok":
                local_items = parsed.get("items", [])
            elif status in ("empty", "need_refine", "error"):
                # stash server message into state so UI shows something useful
                msg = parsed.get("message") or "No results found."
                return {
                    **state,
                    "email_context": {"search_query": args, "search_results": msg, "raw_items": []},
                    "current_task": "searched",
                    "active_account": account,
                }
        except Exception:
            pass

        if local_items:
            # Frame a neat answer for the user from the local results
            def _render_results(items):
                if not items: return "No results."
                if len(items) == 1:
                    i = items[0]
                    dt = i.get("received_at") or ""
                    return f"1 match:\n• {i.get('subject','(no subject)')} — {i.get('sender','?')} — {dt}"
                lines = ["Top results:"]
                for idx, i in enumerate(items[:5], start=1):
                    dt = i.get("received_at") or ""
                    lines.append(f"{idx}. {i.get('subject','(no subject)')} — {i.get('sender','?')} — {dt}")
                return "\n".join(lines)

            rendered = _render_results(local_items)
            return {
                **state,
                "email_context": {"search_query": args, "search_results": rendered, "raw_items": local_items},
                "current_task": "searched","active_account": account,
            }

        if not local_items:
            # no Gmail fallback anymore — DB is the source of truth
            messages = state["messages"] + [AIMessage(content="I couldn’t find matches in the local mail store.")]
            return {**state, "messages": messages, "current_task": "searched", "active_account": account}

    # [REPLACE lines 311–381]
    async def summarize_emails(self, state: AgentState):
        """
        Summarize one OR many emails based on last search results and the user's wording.
        - If user says "summaries", "all", or "top N": summarize multiple (concurrently).
        - Else: keep previous single-email behavior.
        """
        import re, asyncio

        # 0) Pull search context & account
        ctx = state.get("email_context") or {}
        items = (ctx.get("raw_items") or []).copy()
        account = (ctx.get("search_query") or {}).get("account_email") or state.get("active_account") or "primary"

        # 1) If we don’t have items yet, re-run last search with a concrete limit
        if not items:
            last_args = (ctx.get("search_query") or {}).copy()
            if not last_args:
                messages = state["messages"] + [AIMessage(content="No prior search context. Please search for emails first.")]
                return {**state, "messages": messages, "current_task": "summarized"}

            # If the user wants summaries (possibly plural) but we have no items, default to 5
            last_args["limit"] = 5
            rag_json = await self.mcp_client.call_tool("rag_search_emails", last_args)
            try:
                p = json.loads(rag_json or "{}")
                status = p.get("status")
                if status == "one":
                    items = [p.get("item") or {}]
                elif status == "ok":
                    items = p.get("items", [])
                else:
                    msg = p.get("message") or "I couldn’t find emails to summarize."
                    messages = state["messages"] + [AIMessage(content=msg)]
                    return {**state, "messages": messages, "current_task": "summarized"}
            except Exception:
                messages = state["messages"] + [AIMessage(content="Unexpected response from search; cannot select emails.")]
                return {**state, "messages": messages, "current_task": "summarized"}

        if not items:
            messages = state["messages"] + [AIMessage(content="No emails to summarize.")]
            return {**state, "messages": messages, "current_task": "summarized"}

        # 2) Decide how many to summarize from the user's latest message
        user_text = (state["messages"][-1].content or "").lower()
        want_all = any(w in user_text for w in ["all mails", "all emails", "all of them"])
        m_top = re.search(r"\btop\s+(\d{1,3})\b", user_text)
        m_num = re.search(r"\b(\d{1,3})\s+(emails|mails)\b", user_text)
        plural = any(w in user_text for w in ["summaries", "multiple", "many", "list"])

        if want_all:
            k = min(len(items), 25)   # safety cap
        elif m_top:
            k = max(1, min(int(m_top.group(1)), len(items)))
        elif m_num:
            k = max(1, min(int(m_num.group(1)), len(items)))
        elif plural:
            k = min(10, len(items))   # reasonable default for UX
        else:
            k = 1  # default single-email summary

        targets = items[:k]

        # 3) Build payloads (prefer integer id; fallback to thread_id; else allow string id pass-through)
        payloads = []
        for t in targets:
            eid = t.get("email_id")
            tid = t.get("thread_id") or t.get("threadId")
            pl = {"account_email": account}
            if isinstance(eid, int):
                pl["email_id"] = eid
            elif isinstance(eid, str) and eid.isdigit():
                pl["email_id"] = int(eid)
            elif tid:
                pl["thread_id"] = tid
            else:
                # Allow non-numeric Gmail IDs to pass; server has fallback logic
                gid = t.get("gmail_id") or t.get("gmailId") or t.get("external_id") or t.get("email_id")
                if gid:
                    pl["email_id"] = gid
                else:
                    continue
            payloads.append(pl)

        if not payloads:
            messages = state["messages"] + [AIMessage(content="Couldn’t find usable identifiers for summarization.")]
            return {**state, "messages": messages, "current_task": "summarized"}

        # 4) Summarize concurrently to keep latency low
        async def _one(pl):
            try:
                return await self.mcp_client.call_tool("summarize_email", pl)
            except Exception as e:
                return f"Error summarizing one email: {e}"

        results = await asyncio.gather(*[_one(pl) for pl in payloads])

        if k == 1:
            # Single output: behave like before
            messages = state["messages"] + [AIMessage(content=results[0])]
            return {**state, "messages": messages, "current_task": "summarized"}

        # 5) Multi-output: number them to be readable
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}) {r}")

        messages = state["messages"] + [AIMessage(content="\n\n".join(lines))]
        return {**state, "messages": messages, "current_task": "summarized"}

    # [MODIFIED] - Now saves the draft_id and asks for confirmation
    async def draft_emails(self, state: AgentState):
        user_message = state["messages"][-1].content
        # ... (prompts for recipient, subject, body remain the same)
        recipient_prompt = (
            "What is the intended recipient email address (just the email, nothing else) in the following request? "
            f"{user_message}\nRecipient:"
        )
        subject_prompt = (
            "Based on this message, propose a suitable subject line:\n"
            f"{user_message}\nSubject:"
        )
        body_prompt = (
            "Write the body text for an email based on this user instruction:\n"
            f"{user_message}\nBody:"
        )
        to_response = await self.llm.ainvoke(recipient_prompt)
        recipient = to_response.content.strip()
        subject_response = await self.llm.ainvoke(subject_prompt)
        subject = subject_response.content.strip()
        body_response = await self.llm.ainvoke(body_prompt)
        body = body_response.content.strip()

        draft_id = await self.mcp_client.call_tool("draft_email", {
            "recipient": recipient, "subject": subject, "body": body
        })

        # Check if drafting failed
        if "Failed" in draft_id or "Error" in draft_id:
            messages = state["messages"] + [AIMessage(content=draft_id)]
            return {**state, "messages": messages, "draft_id": None}

        confirmation_message = f"I have created a draft for {recipient}. Should I send it?"
        messages = state["messages"] + [AIMessage(content=confirmation_message)]
        return {
            **state,
            "current_task": "drafted",
            "messages": messages,
            "draft_id": draft_id  # Save the ID to the state
        }

    # [NEW] - Node to send the email
    async def send_email(self, state: AgentState):
        draft_id = state.get("draft_id")
        if not draft_id:
            response_text = "I'm sorry, I don't have a draft to send. Please create one first."
            messages = state["messages"] + [AIMessage(content=response_text)]
            return {**state, "messages": messages}

        result = await self.mcp_client.call_tool("send_email", {"draft_id": draft_id})
        messages = state["messages"] + [AIMessage(content=result)]
        return {
            **state,
            "messages": messages,
            "current_task": "sent",
            "draft_id": None  # Clear the ID after sending
        }

    async def generate_response(self, state: AgentState):
        # ... (this function remains unchanged)
        # Check if there are search results to show
        ctx = state.get("email_context", {})
        if ctx.get("search_results"):
            response_text = ctx["search_results"]
        else:
            response_text = "Task complete. What would you like to do next?"
        messages = state["messages"] + [AIMessage(content=response_text)]
        return {
            **state,
            "messages": messages,
            "current_task": "responded"
        }

    async def route_after_search(self, state: AgentState):
        # ... (this function remains unchanged)
        intent = state.get("user_intent", "")
        if "SUMMARIZE" in intent:
            return "summarize"
        else:
            return "respond"


async def run_simple_agent():
    mcp_client = EmailMCPClient()
    try:
        print("Connecting to MCP server...")
        await mcp_client.connect("email_server.py")
        print("Connected to MCP server.")
        agent = GmailAgent(mcp_client, llm)
        print("Gmail Agent started. Type your requests (or 'quit' to exit):")
        compiled_graph = agent.graph.compile()

        # [MODIFIED] - State is now preserved outside the loop
        state = {
            "messages": [],
            "current_task": "",
            "email_context": {},
            "user_intent": "",
            "draft_id": None,
            "active_account": "primary",
        }

        while True:
            user_input = input("\nYour request: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            # Append the new user message to the existing list of messages
            state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

            # Invoke the graph with the updated state
            result = await compiled_graph.ainvoke(state)

            # The result of the graph run becomes the new state for the next turn
            state = result

            messages = result.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                print(f"\nAgent: {messages[-1].content}")
            else:
                print("\nAgent: (No response generated)")
    finally:
        try:
            await mcp_client.exit_stack.aclose()
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(run_simple_agent())