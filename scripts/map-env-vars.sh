# echo "Loading azd .env file from current environment..."

# while IFS='=' read -r key value; do
#     value=$(echo "$value" | sed 's/^"//' | sed 's/"$//')
#     export "$key=$value"
# done <<EOF
# $(azd env get-values)
# EOF

azd env get-values > src/.env
echo "azd .env file loaded successfully."