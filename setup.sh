mkdir -p ~/.streamlit/
wget https://github.com/CVxTz/TagSuggestionImages/releases/download/v0.1/labels.json
wget https://github.com/CVxTz/TagSuggestionImages/releases/download/v0.1/image_representation.h5
echo "\
[general]\n\
email = \"mansaryounessecp@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml