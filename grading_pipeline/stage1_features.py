import os
import sys
import io
import json
import sqlite3
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# =====================================================
# CONFIGURATION & API KEYS
# =====================================================
os.environ["GEMINI_API"] = ""
os.environ["HF_TOKEN"] = ""

# Hugging Face Runtime Adjustments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

SENDER_EMAIL = ""
SENDER_PASSWORD = ""

# Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =====================================================
# PATHS
# =====================================================
BASE_PATH = r"C:\AI_Grading_System"
DB_DIR = os.path.join(BASE_PATH, "database")
DB_PATH = os.path.join(DB_DIR, "gibo_master.db")

INPUT_JSON = os.path.join(BASE_PATH, "grading_engine", "matched_answers.json")
OUTPUT_JSON = os.path.join(BASE_PATH, "grading_engine", "stage1_features.json")
STAGE2_JSON = os.path.join(BASE_PATH, "grading_engine", "stage2_marks_allocation.json")
STAGE3_JSON = os.path.join(BASE_PATH, "grading_engine", "stage3_results.json")
RUBRIC_CSV = os.path.join(BASE_PATH, "dataset", "Machine_Learning", "metadata", "master_rubric.csv")
RUBRIC_JSON = os.path.join(BASE_PATH, "dataset", "Machine_Learning", "metadata", "rubric.json")
STUDENT_DETAILS_CSV = os.path.join(BASE_PATH, "dataset", "Machine_Learning", "metadata", "students_details.csv")

# =====================================================
# DATABASE UTILITIES
# =====================================================
def get_connection():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS student_details")
    cur.execute("DROP TABLE IF EXISTS students_details")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS students_details (
        ans_id TEXT PRIMARY KEY, 
        template_id TEXT, 
        roll_no TEXT, 
        name TEXT, 
        email TEXT, 
        class TEXT)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS grading_results (
        ans_id TEXT PRIMARY KEY, name TEXT, roll_no TEXT, template_id TEXT, 
        marks_awarded REAL, total_marks REAL, percentage TEXT, grade TEXT, transparency_json TEXT)""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS gibo_evaluations (
        ans_id TEXT PRIMARY KEY, name TEXT, roll_no TEXT, template_id TEXT, 
        gibo_intro TEXT, email_report TEXT)""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS gibo_detailed_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ans_id TEXT, qid TEXT, 
        feedback TEXT, missing_concepts TEXT, mini_lesson TEXT)""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS marks_per_question (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ans_id TEXT, section TEXT, 
        qid TEXT, status TEXT, marks_obtained REAL, max_marks REAL)""")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS master_rubric (
        template_id TEXT, qid TEXT, question_text TEXT, marks REAL, 
        ideal_reference_answer TEXT, key_pillars TEXT, PRIMARY KEY (template_id, qid))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS rubric_templates (
        template_id TEXT, qid TEXT, original_label TEXT, section TEXT, 
        question_text TEXT, marks REAL, keywords TEXT, PRIMARY KEY (template_id, qid))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS student_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ans_id TEXT, qid TEXT, 
        semantic_score REAL, logic_score REAL, actual_word_count INTEGER, 
        target_word_count INTEGER, length_compliance_score REAL, keyword_coverage REAL)""")
    
    cur.execute("CREATE TABLE IF NOT EXISTS gibo_quota (date TEXT PRIMARY KEY, usage_count INTEGER DEFAULT 0)")
    cur.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT, role TEXT)")

    cur.execute("""CREATE TABLE IF NOT EXISTS teacher_table (
        email TEXT PRIMARY KEY, 
        name TEXT, 
        mobile TEXT, 
        password TEXT,
        status TEXT DEFAULT 'pending'
    )""")

    conn.commit()
    conn.close()
    print("✨ Database structures ready.")

def sync_data():
    conn = get_connection()
    print("🔄 Starting Sync...")

    conn.execute("DELETE FROM students_details WHERE length(ans_id) <= 5")
    conn.execute("DELETE FROM grading_results WHERE length(ans_id) <= 5")
    conn.execute("DELETE FROM gibo_evaluations WHERE length(ans_id) <= 5")
    conn.commit()

    if os.path.exists(STUDENT_DETAILS_CSV):
        try:
            df_details = pd.read_csv(STUDENT_DETAILS_CSV).dropna(how='all')
            df_details.columns = [c.strip().lower() for c in df_details.columns]
            
            for _, row in df_details.iterrows():
                raw_id = str(row.get('ans_id', '')).strip()
                if raw_id.startswith('T') and len(raw_id) <= 5:
                    continue 

                conn.execute("""INSERT OR REPLACE INTO students_details 
                    (ans_id, template_id, roll_no, name, email, class) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (raw_id, str(row.get('template_id', '')), 
                     str(row.get('roll_no', '')), str(row.get('name', '')), 
                     str(row.get('email', '')), str(row.get('class', ''))))
            conn.commit()
            print("✅ Student details synced.")
        except Exception as e: 
            print(f"❌ Student Details Sync Error: {e}")

    if os.path.exists(RUBRIC_CSV):
        try:
            df = None
            for enc in ['utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(RUBRIC_CSV, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df is not None:
                df.columns = [c.strip().lower() for c in df.columns]
                df = df.astype(str).replace('nan', '')
                for _, row in df.iterrows():
                    conn.execute("""INSERT OR REPLACE INTO master_rubric 
                        (template_id, qid, question_text, marks, ideal_reference_answer, key_pillars) 
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (row.get('template_id', '').strip(), row.get('qid', '').strip(), 
                         row.get('question_text', '').strip(), row.get('marks', '0.0'), 
                         row.get('ideal_reference_answer', '').strip(), row.get('key_pillars', '').strip()))
                conn.commit()
                print("✅ master_rubric synced.")
        except Exception as e: 
            print(f"❌ CSV Sync Error: {e}")

    if os.path.exists(RUBRIC_JSON):
        try:
            with open(RUBRIC_JSON, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for tid, q_list in raw_data.items():
                    for q in q_list:
                        conn.execute("""INSERT OR REPLACE INTO rubric_templates 
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (tid, q.get('qid'), q.get('original_label'), q.get('section'), 
                             q.get('text'), float(q.get('marks', 0.0)), ", ".join(q.get('keywords', []))))
            conn.commit()
            print("✅ rubric_templates updated.")
        except Exception as e: 
            print(f"❌ JSON Rubric Error: {e}")

    if os.path.exists(STAGE2_JSON):
        try:
            with open(STAGE2_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for aid, content in data.items():
                if not aid.startswith('ans_'): continue 

                info = content.get('student_info', {})
                res = content.get('final_result', {})
                report = content.get('transparency_report', {})
                
                awarded = float(res.get('marks_awarded', 0.0))
                total = float(res.get('total_marks', 0.0))
                perc_val = res.get('percentage', f"{(awarded/total*100):.1f}%" if total > 0 else "0%")
                grade_val = res.get('grade', "F")

                conn.execute("""INSERT OR REPLACE INTO grading_results 
                    (ans_id, name, roll_no, template_id, marks_awarded, total_marks, percentage, grade, transparency_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (aid, info.get('name'), info.get('roll_no'), info.get('template_id'),
                     awarded, total, perc_val, grade_val, json.dumps(report)))
                
                conn.execute("DELETE FROM marks_per_question WHERE ans_id=?", (aid,))
                
                for section_name, section_data in report.items():
                    if isinstance(section_data, dict):
                        for q in section_data.get('question_breakdown', []):
                            qid = q.get('qid')
                            status = q.get('status')
                            raw_marks = q.get('marks', "0.0 / 0.0")
                            try:
                                m_obt = float(raw_marks.split('/')[0].strip())
                                m_max = float(raw_marks.split('/')[1].strip())
                            except Exception:
                                m_obt, m_max = 0.0, 0.0

                            conn.execute("""INSERT INTO marks_per_question 
                                (ans_id, section, qid, status, marks_obtained, max_marks) 
                                VALUES (?, ?, ?, ?, ?, ?)""",
                                (aid, section_name, qid, status, m_obt, m_max))
            
            conn.commit()
            print(f"✅ Synced stage 2 records.")
        except Exception as e:
            print(f"❌ Stage 2 Sync Error: {e}")

    if os.path.exists(STAGE3_JSON):
        try:
            with open(STAGE3_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for aid, content in data.items():
                    if not aid.startswith('ans_'): continue
                    info, gibo = content.get('student_info', {}), content.get('gibo_letter', {})
                    conn.execute("INSERT OR REPLACE INTO gibo_evaluations VALUES (?, ?, ?, ?, ?, ?)",
                                 (aid, info.get('name'), info.get('roll_no'), info.get('template_id'), gibo.get('intro_message', ''), content.get('email_report', '')))
                    
                    conn.execute("DELETE FROM gibo_detailed_feedback WHERE ans_id=?", (aid,))
                    for item in gibo.get('results', []):
                        conn.execute("INSERT INTO gibo_detailed_feedback (ans_id, qid, feedback, missing_concepts, mini_lesson) VALUES (?, ?, ?, ?, ?)",
                                     (aid, item.get('qid'), item.get('feedback'), ", ".join(item.get('missing_concepts', [])), item.get('mini_lesson')))
            conn.commit()
            print("✅ Stage 3 Feedback synced.")
        except Exception as e: 
            print(f"❌ Stage 3 Sync Error: {e}")
    
    conn.close()

# =====================================================
# FEATURE EXTRACTION ENGINE
# =====================================================
def get_target_word_count(marks):
    m = float(marks)
    if m <= 2: return 70
    elif m <= 3: return 90
    elif m <= 5: return 150
    elif m <= 8: return 170
    elif m <= 10: return 200
    else: return 250

def run_feature_extraction():
    if not os.path.exists(RUBRIC_CSV): 
        print("Error: Master Rubric missing")
        return
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found")
        return

    rubric_df = pd.read_csv(RUBRIC_CSV, encoding="latin-1")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_matched_data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Loading AI Models on {device}...")
    
    try:
        ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        depth_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print("✅ Models Loaded Successfully!")
    except Exception as e:
        print(f"❌ Model Loading Failed: {e}")
        return

    final_features = {}
    total_students = len(all_matched_data)
    
    print(f"🚀 Starting Feature Extraction for {total_students} students...")
    
    for idx, (sid, content) in enumerate(all_matched_data.items(), 1):
        print(f"[{idx}/{total_students}] Processing: {sid}")
        
        tid = content['student_info'].get('template_id')
        analysis_list = []

        for item in content['rubric_with_answers']:
            ans_text = str(item.get('student_answer', ""))
            qid = item.get('qid')
            max_marks = float(item.get('max_marks', 0))
            target_goal = get_target_word_count(max_marks)
            actual_words = len(ans_text.split())
            
            logic_score = 0.0
            semantic_score = 0.0
            final_grade_score = 0.0
            
            if "SKIPPED" in ans_text or actual_words < 15:
                pass
            else:
                ref_match = rubric_df[(rubric_df['template_id'] == tid) & (rubric_df['qid'] == qid)]
                if not ref_match.empty:
                    ideal_ans = str(ref_match.iloc[0]['ideal_reference_answer'])
                    
                    raw_logic = ranker.predict((ideal_ans, ans_text))
                    logic_score = float(1 / (1 + pow(2.718, -raw_logic))) 

                    emb1 = depth_model.encode(ideal_ans, convert_to_tensor=True)
                    emb2 = depth_model.encode(ans_text, convert_to_tensor=True)
                    depth_val = float(util.pytorch_cos_sim(emb1, emb2))
                    semantic_score = depth_val

                    final_grade_score = (logic_score * 0.6) + (depth_val * 0.4)

            item.update({
                "semantic_score": round(final_grade_score, 4),
                "logic_score": round(logic_score, 4),
                "depth_score": round(semantic_score, 4),
                "actual_word_count": actual_words,
                "target_word_count": target_goal,
                "length_compliance_score": min(round(actual_words / target_goal, 4), 1.0) if target_goal > 0 else 0.0
            })
            analysis_list.append(item)

        final_features[sid] = {
            "info": content['student_info'], 
            "analysis": analysis_list
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    print(f"💾 Saving all features to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_features, f, indent=4, ensure_ascii=False)
    
    print("✅ Done! All student features extracted.")

# =====================================================
# AUTHENTICATION & TEACHER MANAGEMENT
# =====================================================
def register_teacher(name, email, mobile, _):
    try:
        conn = get_connection()
        fixed_pass = "TeacherLogin" 
        
        conn.execute("""
            INSERT INTO teacher_table (email, name, mobile, password, status) 
            VALUES (?, ?, ?, ?, 'pending')
        """, (email, name, mobile, fixed_pass))
        conn.commit()
        conn.close()
        return True, "request sent ! check your email"
    except Exception:
        return False, "This email is already registered."

def hard_delete_teacher(email):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE email = ?", (email,))
    conn.execute("DELETE FROM teacher_table WHERE email = ?", (email,))
    conn.commit()
    conn.close()
    return True

def verify_login(username_email, password):
    if username_email == "admin" and password == "admin123":
        return True, {"email": "admin", "role": "admin"}
    if username_email == "teacher" and password == "teacher123":
        return True, {"email": "teacher", "role": "teacher"}

    try:
        conn = get_connection()
        user = conn.execute("SELECT email, role FROM users WHERE email = ? AND password = ?", 
                            (username_email, password)).fetchone()
        conn.close()

        if user:
            return True, {"email": user[0], "role": user[1]}
        return False, None
    except Exception:
        return False, None

def send_approval_email(receiver_email, password):
    subject = "🎉 Your Teacher Account is Approved!"
    body = f"""Hello,

Your registration for the AI Grading System has been approved by the Admin.

You can now login using these credentials:
Username: {receiver_email}
Password: {password}

Best Regards,
GIBO AI Team"""
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

def approve_teacher_request(email):
    try:
        conn = get_connection()
        teacher = conn.execute("SELECT email, password FROM teacher_table WHERE email = ?", (email,)).fetchone()
        
        if teacher:
            conn.execute("INSERT OR REPLACE INTO users (email, password, role) VALUES (?, ?, 'teacher')", 
                         (teacher[0], teacher[1]))
            conn.execute("UPDATE teacher_table SET status = 'approved' WHERE email = ?", (email,))
            conn.commit()
            conn.close()

            send_approval_email(teacher[0], teacher[1])
            return True
        return False
    except Exception:
        return False

def get_all_teacher_requests():
    conn = get_connection()
    df = pd.read_sql_query("SELECT name, email, mobile, status FROM teacher_table", conn)
    conn.close()
    return df

def block_teacher_request(email):
    conn = get_connection()
    conn.execute("UPDATE teacher_table SET status = 'blocked' WHERE email = ?", (email,))
    conn.execute("DELETE FROM users WHERE email = ?", (email,))
    conn.commit()
    conn.close()

def get_class_missing_concepts(template_id):
    try:
        conn = get_connection()
        query = """
            SELECT f.missing_concepts 
            FROM gibo_detailed_feedback f
            JOIN grading_results r ON f.ans_id = r.ans_id
            WHERE r.template_id = ? AND f.missing_concepts != ''
        """
        df = pd.read_sql_query(query, conn, params=(template_id,))
        conn.close()
        
        if not df.empty:
            concepts = df['missing_concepts'].str.split(',').explode().str.strip()
            return concepts.value_counts().to_dict()
        return {}
    except Exception as e:
        print(f"Error fetching concepts: {e}")
        return {}

def update_manual_grade(ans_id, new_marks, new_grade):
    try:
        conn = get_connection()
        conn.execute("""
            UPDATE grading_results 
            SET marks_awarded = ?, grade = ? 
            WHERE ans_id = ?
        """, (new_marks, new_grade, ans_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating grade: {e}")
        return False

# =====================================================
# MAIN ENTRY POINT
# =====================================================
if __name__ == "__main__":
    init_db()
    sync_data()
    run_feature_extraction()