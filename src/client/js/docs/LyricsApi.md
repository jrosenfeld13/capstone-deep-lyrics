# LyricsGeneration.LyricsApi

All URIs are relative to *https://localhost:8080*

Method | HTTP request | Description
------------- | ------------- | -------------
[**generateLyrics**](LyricsApi.md#generateLyrics) | **POST** /lyrics | Request lyric generation.


<a name="generateLyrics"></a>
# **generateLyrics**
> &#39;String&#39; generateLyrics(body)

Request lyric generation.

Request lyric generation.

### Example
```javascript
var LyricsGeneration = require('lyrics_generation');

var apiInstance = new LyricsGeneration.LyricsApi();

var body = new LyricsGeneration.LyricsRequest(); // LyricsRequest | LyricsRequest object that defines parameters of the lyrics desired.


var callback = function(error, data, response) {
  if (error) {
    console.error(error);
  } else {
    console.log('API called successfully. Returned data: ' + data);
  }
};
apiInstance.generateLyrics(body, callback);
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**LyricsRequest**](LyricsRequest.md)| LyricsRequest object that defines parameters of the lyrics desired. | 

### Return type

**&#39;String&#39;**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

