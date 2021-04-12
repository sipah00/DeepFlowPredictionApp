mkdir -p ~/.streamlit/

echo $PORT

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
