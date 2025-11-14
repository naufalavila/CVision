import streamlit as st
import PyPDF2
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()

st.set_page_config(page_title='CVision', layout='centered')
st.image("logo_cvision.png", width=400)
st.markdown('### Your AI Resume Reviewer!')
st.markdown('#### A platform where you can get to know about your resume deeply.')
st.markdown('Wanna improve your CV? Upload a PDF file and let AI do the work to analyze it!')

# === Load API Keys ===
LLAMA_338B_INSTRUCT = os.getenv('LLAMA_338B_INSTRUCT')
NVIDIA_NEMOTRON_NANO = os.getenv('NVIDIA_NEMOTRON_NANO')
GOOGLE_GEMMA_3 = os.getenv('GOOGLE_GEMMA_3')
LLAMA_4_SCOUT = os.getenv('LLAMA_4_SCOUT')
GPT_OSS_20B = os.getenv('GPT_OSS_20B')

# === File Upload ===
uploaded_file = st.file_uploader("Upload PDF", type='pdf')

# === Select Language ===
output_language = st.selectbox(
    "Choose Language for Analysis:",
    ["Bahasa Indonesia", "English"]
)

# === Dropdown Menu Model ===
model_choice = st.selectbox(
    "Choose AI Model:",
    [
        "meta-llama/llama-3.3-8b-instruct",
        "nvidia/nemotron-nano-12b-v2-vl",
        "google/gemma-3-27b-it",
        "meta-llama/llama-4-scout",
        "openai/gpt-oss-20b"
    ]
)

# === Tombol Analisis ===
analyze = st.button('Analyze')

# === Fungsi Ekstraksi Text ===
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

# === Model Config Berdasarkan Pilihan ===
def get_model_config(choice):
    if choice == "meta-llama/llama-3.3-8b-instruct":
        return {
            "api_key": LLAMA_338B_INSTRUCT,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.3-8b-instruct:free"
        }
    elif choice == "nvidia/nemotron-nano-12b-v2-vl":
        return {
            "api_key": NVIDIA_NEMOTRON_NANO,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "nvidia/nemotron-nano-12b-v2-vl:free"
        }
    elif choice == "google/gemma-3-27b-it":
        return {
            "api_key": GOOGLE_GEMMA_3,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemma-3-27b-it:free"
        }
    elif choice == "meta-llama/llama-4-scout":
        return {
            "api_key": LLAMA_4_SCOUT,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-4-scout:free"
        }
    elif choice == "openai/gpt-oss-20b":
        return {
            "api_key": GPT_OSS_20B,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-oss-20b:free"
        }
    else:
        return None
    
#######################################################
def create_pdf(content):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x, y = 50, height - 50
    text_object = c.beginText(x, y)
    text_object.setFont("Helvetica", 11)

    # Pisahkan teks panjang menjadi baris-baris
    for line in content.split("\n"):
        text_object.textLine(line)
        if text_object.getY() < 50:  # Jika sudah sampai bawah halaman
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(x, height - 50)
            text_object.setFont("Helvetica", 11)

    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer

# === Proses Analisis ===
if analyze and uploaded_file:
    try:
        config = get_model_config(model_choice)

        if not config or not config["api_key"]:
            st.error("API key untuk model ini belum diset di file .env kamu.")
            st.stop()

        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("File tidak memiliki konten yang bisa dibaca.")
            st.stop()

        prompt = f"""
        Analisislah CV atau resume di bawah ini dan berikan masukan yang membangun.
        Fokus pada hal-hal berikut:
        1. Kejelasan dan dampak isi CV
        2. Penyajian keterampilan (skills)
        3. Deskripsi pengalaman kerja
        4. Saran spesifik agar lebih sesuai dengan posisi yang dilamar pada dokumen {uploaded_file}

        Hasil analisis harus berupa 5 poin berikut:
        1. Penjelasan singkat tentang CV tersebut
        2. Kekurangan (dalam bentuk poin bernomor)
        3. Kelebihan (dalam bentuk poin bernomor)
        4. Rekomendasi perbaikan agar lebih profesional dan menarik.

        Gunakan bahasa: {output_language}. Buatlah tulisannya menjadi friendly dan mudah dipahami.

        Konten CV:
        {file_content}
        """

        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

        with st.spinner(f"Analyzing..."):
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "You are an experienced HR and resume reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

        # === Result Output===
        analysis_result = response.choices[0].message.content
        st.markdown("### Hasil Analisis")
        st.markdown(analysis_result)

        # === Tombol Download muncul hanya jika ada hasil ===
        pdf_buffer = create_pdf(analysis_result)
        st.download_button(
            label="Download Analysis (PDF)",
            data=pdf_buffer,
            file_name=f"hasil_analisis_{uploaded_file.name.replace('.pdf', '')}.pdf",
            mime="application/pdf"
        ) 

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
