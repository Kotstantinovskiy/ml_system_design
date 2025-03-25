from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "FacebookAI/xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define language labels based on the model's training
LANGUAGE_LABELS = [
    'Acoli', 'Adangme', 'Adhola', 'Tunisian Arabic', 'Afrikaans', 'Alur', 'Amharic', 'Anuak', 
    'Arabic', 'Arabic with Diacritics', 'Assamese', 'Bemba', 'Bengali', 'Tibetan', 'Bukusu', 
    'Catalan', 'Chopi', 'Central Kurdish (Sorani)', 'Plains Cree', 'Swampy Cree', 'Chol', 
    'Danish', 'Dagbani', 'German', 'Dagaare', 'Dinka', 'Zarma', 'Ewe', 'Greek', 'English', 
    'Esperanto', 'Spanish', 'Persian', 'Persian with Diacritics', 'Fante', 'Fulah', 'French', 
    'Ga', 'Gonja', 'Gujarati', 'Farefare', 'Gusii', 'Guyanese Creole English', 'Hausa', 
    'Serbo-Croatian', 'Huichol', 'Hindi', 'Haitian Creole', 'Hungarian', 'Huastec', 'Herero', 
    'Indonesian', 'Italian', 'Japanese', 'Jamaican Creole English', 'Kamba', 'Karamojong', 
    'Kakwa', 'Kikuyu', 'Kuanyama', 'Kalenjin', 'Khmer', 'Northern Kurdish (Kurmanji)', 'Kannada', 
    'Korean', 'Konkani', 'Konzo', 'Kupsabiny', 'Kaonde', 'Kanuri', 'Krio', 'Kurukh', 'Juǀʼhoan', 
    'Kwangali', 'Latin', 'Lango', 'Ganda', 'Lugbara', 'Lugbara (Official Orthography)', 'Khayo', 
    'Lingala', 'Lozi', 'Lozi (Namibia)', 'Lozi (Zambia)', 'Saamia', 'Lithuanian', 'Aringa', 
    'Luvale', 'Lunda', 'Luo', 'Wanga', 'Maasai', 'Matumbi', 'Central Mazahua', 'Meru', 'Morisyen', 
    'Malagasy', 'Maʼdi', 'Mbukushu', 'Mikir', 'Malayalam', 'Michoacán Mazahua', 'Mon', 'Mandari', 
    'Marathi', 'Malay', 'Burmese', 'Masaaba', 'Nama', 'Norwegian Bokmål', 'Central Huasteca Nahuatl', 
    'Nepali', 'Ndonga', 'Eastern Huasteca Nahuatl', 'Western Huasteca Nahuatl', 'Dutch', 'East Nyala', 
    'Norwegian Nynorsk', 'Norwegian', 'Norwegian (IPA)', 'Southern Ndebele', 'Northern Sotho', 'Nyole', 
    'Nyanja', 'Nyankole', 'Nyungwe', 'Nzima', 'Atzingo Matlatzinca', 'Mochi', 'Oromo', 'Odia (Oriya)', 
    'Punjabi', 'Punjabi (Shahmukhi script)', 'Nigerian Pidgin', 'Polish', 'Northern Pame', 'Dari', 
    'Dari with Diacritics', 'Pashto', 'Portuguese', 'Rakhine', 'Romanian', 'Russian', 'Kinyarwanda', 
    'Sanskrit', 'Samburu', 'Sadri', 'Northern Sami', 'Sango', 'Somali', 'Albanian', 'Serbian', 'Swati', 
    'Southern Sotho', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Teso', 'Tetum', 'Thai', 'Tigrinya', 
    'Tagalog', 'Tswana', 'Tohono Oʼodham', 'Tonga (Zambia)', 'Turkish', 'Tsonga', 'Tswa', 'Tooro', 
    'Tumbuka', 'Turkana', 'Twi (Akuapem)', 'Twi (Asante)', 'Ukrainian', 'Urdu', 'Venda', 'Vietnamese', 
    'Xhosa', 'Soga', 'Kasem', 'Yoruba', 'Yucateco', 'Cantonese', 'Chinese', 'Chinese (Pinyin)', 
    'Zande', 'Zulu'
]



import streamlit as st

# Streamlit UI
st.title("Language Identification")
st.write("Enter text below to identify its language:")

# Text input
user_input = st.text_area("Your text here", "")

if st.button("Identify Language"):
    if user_input:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_label].item()

        # Display result
        st.write(f"**Predicted Language:** {LANGUAGE_LABELS[predicted_label]}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Please enter some text to identify.")