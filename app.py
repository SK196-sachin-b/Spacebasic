import streamlit as st
import sys
import os
import uuid

# Fix imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "service"))

st.set_page_config(page_title="SpaceBasic RAG Chatbot", layout="wide")

# ✅ SESSION LOADING FUNCTION
def load_session(session_id):
    """Load a specific session and restore chat history"""
    try:
        from db import db
        if db.connect():
            # Check if session exists, if not create it
            db.cursor.execute("SELECT session_id FROM chat_sessions WHERE session_id = %s;", (session_id,))
            session_exists = db.cursor.fetchone()
            
            if not session_exists:
                # Create the session if it doesn't exist
                db.cursor.execute("INSERT INTO chat_sessions (session_id) VALUES (%s);", (session_id,))
                db.connection.commit()
                print(f"✅ Created missing session: {session_id}")
            
            history = db.get_chat_history(session_id, limit=1000)  # Load ALL messages for UI
            db.close()
            
            # Update session state
            st.session_state.session_id = session_id
            st.session_state.messages = []
            
            # Convert DB history to Streamlit format
            for h in history:
                st.session_state.messages.append({
                    "role": h["role"],
                    "content": h["message"]
                })
            
            print(f"✅ Loaded session {session_id} with {len(history)} messages")
            return True
        else:
            print("❌ Failed to connect to database")
            return False
    except Exception as e:
        print(f"❌ Error loading session: {e}")
        return False

# -------------------------------
# LOAD SERVICE
# -------------------------------
@st.cache_resource
def load_qa_service():
    from qa import qa_service
    return qa_service

qa_service = load_qa_service()

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # ✅ SESSION MANAGEMENT
    st.subheader("💬 Chat Sessions")
    
    # Initialize session_id if not exists
    if "session_id" not in st.session_state:
        # Create session in database
        from db import db
        if db.connect():
            session_id = db.create_session()
            db.close()
            if session_id:
                st.session_state.session_id = session_id
                print(f"🆕 Created new session in DB: {session_id}")
            else:
                # Fallback to UUID if DB creation fails
                st.session_state.session_id = str(uuid.uuid4())
                print(f"⚠️ DB session creation failed, using UUID: {st.session_state.session_id}")
        else:
            # Fallback to UUID if DB connection fails
            st.session_state.session_id = str(uuid.uuid4())
            print(f"⚠️ DB connection failed, using UUID: {st.session_state.session_id}")
    
    # Show current session
    st.write(f"**Current Session:** {st.session_state.session_id[:8]}...")
    
    # New session button
    if st.button("🆕 New Session"):
        from db import db
        if db.connect():
            new_session_id = db.create_session()
            db.close()
            if new_session_id:
                st.session_state.session_id = new_session_id
                st.session_state.messages = []
                print(f"🆕 Created new session in DB: {new_session_id}")
                st.rerun()
            else:
                st.error("Failed to create new session in database")
        else:
            st.error("Cannot connect to database to create session")
    
    # Load session functionality
    with st.expander("📂 Load Session"):
        try:
            from db import db
            if db.connect():
                sessions = db.get_all_sessions()
                db.close()
                
                if sessions:
                    for session in sessions[:5]:  # Show last 5 sessions
                        session_preview = f"{session['session_id'][:8]}... ({session['message_count']} msgs)"
                        if st.button(session_preview, key=f"load_{session['session_id']}"):
                            load_session(session['session_id'])
                            st.rerun()
                else:
                    st.write("No previous sessions found")
            else:
                st.write("Cannot connect to database")
        except Exception as e:
            st.write(f"Error loading sessions: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if "role" not in st.session_state:
        st.session_state.role = "student"

    st.session_state.role = st.selectbox(
        "Role",
        ["student", "staff"],
        index=["student", "staff"].index(st.session_state.role)
    )

role = st.session_state.role

# -------------------------------
# CHAT UI
# -------------------------------
st.title("🤖 SpaceBasic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        # ✅ Pass session_id to QA service
        response = qa_service.ask(user_input, st.session_state.session_id)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------
# ADMIN PANEL
# -------------------------------
st.divider()
st.subheader("✏️ Admin Panel")

if role != "staff":
    st.info("🔒 Only staff can manage content")
    st.stop()

operation = st.selectbox("Operation", ["Create", "Update", "Delete"])

file_name = st.text_input("File name (.pdf)")
new_content = st.text_area("Content")

confirm = st.checkbox("Confirm operation")

col1, col2 = st.columns(2)
preview_clicked = col1.button("🔍 Preview")
execute_clicked = col2.button("🚀 Execute")

# -------------------------------
# PREVIEW
# -------------------------------
if preview_clicked:
    try:
        from update_service import preview_update, preview_delete_by_content

        if operation == "Update":
            st.session_state.preview_data = preview_update(role, file_name, new_content)

        elif operation == "Delete":
            st.session_state.preview_data = preview_delete_by_content(role, file_name, new_content)

        elif operation == "Create":
            st.session_state.preview_data = {
                "status": "create_ready",
                "file_name": file_name,
                "new_content": new_content,
                "operation": "Create"
            }
    except ImportError:
        st.error("Update service not available")

# -------------------------------
# DISPLAY PREVIEW
# -------------------------------
if "preview_data" in st.session_state:
    preview_data = st.session_state.preview_data

    # ================= UPDATE =================
    if operation == "Update" and preview_data["status"] == "chunks_found":

        st.success(f"✅ Found {len(preview_data['chunks'])} chunks")

        key = f"update_sel_{file_name}"
        ids = {c["id"] for c in preview_data["chunks"]}

        # Fix KeyError issue
        st.session_state[key] = {
            cid: st.session_state.get(key, {}).get(cid, True)
            for cid in ids
        }

        selections = st.session_state[key]

        with st.form("update_form"):

            for c in preview_data["chunks"]:
                st.checkbox(
                    f"Chunk {c['id']} (Score: {c['score']})",
                    value=selections.get(c["id"], True),
                    key=f"upd_{c['id']}"
                )

                with st.expander(f"📄 Content Preview - Chunk {c['id']}"):
                    st.code(c["content_preview"], language="text")

            submitted = st.form_submit_button("✅ Apply Selection")

        if submitted:
            new_sel = {}
            for c in preview_data["chunks"]:
                new_sel[c["id"]] = st.session_state.get(f"upd_{c['id']}", False)

            st.session_state[key] = new_sel
            st.rerun()

        selections = st.session_state[key]
        selected = [c for c in preview_data["chunks"] if selections.get(c["id"], False)]

        st.session_state.preview_data["selected_chunks"] = selected

        if selected:
            st.success(f"✅ {len(selected)} chunks selected")
        else:
            st.warning("⚠️ No chunks selected")

    # ================= DELETE =================
    elif operation == "Delete" and preview_data["status"] == "chunks_found":

        st.success(f"✅ Found {len(preview_data['chunks'])} chunks")

        key = f"delete_sel_{file_name}"
        ids = {c["id"] for c in preview_data["chunks"]}

        st.session_state[key] = {
            cid: st.session_state.get(key, {}).get(cid, False)
            for cid in ids
        }

        selections = st.session_state[key]

        with st.form("delete_form"):

            for c in preview_data["chunks"]:
                st.checkbox(
                    f"Chunk {c['id']} (Score: {c['score']})",
                    value=selections.get(c["id"], False),
                    key=f"del_{c['id']}"
                )

                with st.expander(f"📄 Content Preview - Chunk {c['id']}"):
                    st.code(c["content_preview"], language="text")

            submitted = st.form_submit_button("🗑️ Apply Selection")

        if submitted:
            new_sel = {}
            for c in preview_data["chunks"]:
                new_sel[c["id"]] = st.session_state.get(f"del_{c['id']}", False)

            st.session_state[key] = new_sel
            st.rerun()

        selections = st.session_state[key]
        selected = [c for c in preview_data["chunks"] if selections.get(c["id"], False)]

        st.session_state.preview_data["selected_chunks"] = selected

        if selected:
            st.success(f"✅ {len(selected)} chunks selected for deletion")
        else:
            st.warning("⚠️ No chunks selected")

    # ================= CREATE =================
    elif operation == "Create":
        st.success("✅ Ready to create new content")

# -------------------------------
# EXECUTE
# -------------------------------
if execute_clicked and confirm and "preview_data" in st.session_state:
    try:
        from update_service import confirm_update, confirm_delete, create_content

        data = st.session_state.preview_data

        if operation == "Create":
            result = create_content(role, file_name, new_content)

        elif operation == "Update":
            result = confirm_update(data)

        elif operation == "Delete":
            result = confirm_delete(data)

        if "✅" in result:
            st.success(result)
            del st.session_state.preview_data
        else:
            st.error(result)
    except ImportError:
        st.error("Update service not available")