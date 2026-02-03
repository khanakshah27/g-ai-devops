# Crime Detection API Documentation

This API allows you to upload video files for automated crime detection analysis using a hybrid R3D (3D ResNet) and Random Forest model.

## Base URL

`https://khanak27-crime-detection-api.hf.space`

## Authentication

This API requires an API Key to be sent in the headers for all requests.

| Header | Value | Description |
| :--- | :--- | :--- |
| `x-api-key` | `default-secret-key` | The secret key to authorize the request. |

*(Note: If the server administrator has configured a custom secret, use that instead of the default).*

## Endpoints

### 1. Predict Crime

Analyzes a video file and returns the classification (Normal vs. Crime) and confidence score.

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Request Body (Form Data)

| Key | Type | Description |
| :--- | :--- | :--- |
| `file` | `File` | The video file to be analyzed. Supported formats: `.mp4`, `.mov`. |

#### Example Request (cURL)

```bash
curl -X POST "[https://khanak27-crime-detection-api.hf.space/predict](https://khanak27-crime-detection-api.hf.space/predict)" \
  -H "x-api-key: default-secret-key" \
  -F "file=@/path/to/your/video.mp4"


## Example Response

{

"confidence":"57.00%",
"filename":"good1.mp4",
"is_crime":false,
"prediction":"Normal"

}

