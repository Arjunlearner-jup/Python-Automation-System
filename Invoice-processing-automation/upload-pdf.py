import streamlit as st


st.set_page_config(page_title="PDF Upload", page_icon="📄")
st.title("Upload a PDF")

uploaded_pdf = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()

    if not pdf_bytes.startswith(b"%PDF-"):
        st.error("The selected file does not appear to be a valid PDF.")
    else:
        st.success(f"Uploaded: {uploaded_pdf.name}")
        st.write(f"File size: {len(pdf_bytes):,} bytes")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=uploaded_pdf.name,
            mime="application/pdf",
        )
