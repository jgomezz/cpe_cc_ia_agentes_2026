import streamlit as st

st.title("Agentes con Streamlit")

st.write("Hola Mundo desde Streamlit! 👋")

nombre = st.text_input("¿Cómo te llamas?")
if nombre:
    st.write(f"Hola, {nombre}!")

if st.button("Saludar"):
    st.success("¡Botón presionado!")

edad = st.slider("Tu edad", 0, 100, 25)
st.write(f"Tienes {edad} años")

