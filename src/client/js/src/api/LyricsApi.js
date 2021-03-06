/**
 * Lyrics Generation
 * Lyric generation capstone project.
 *
 * OpenAPI spec version: 1.0.0
 *
 * NOTE: This class is auto generated by the swagger code generator program.
 * https://github.com/swagger-api/swagger-codegen.git
 *
 * Swagger Codegen version: 2.3.1
 *
 * Do not edit the class manually.
 *
 */

(function(root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['ApiClient', 'model/LyricsRequest'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    module.exports = factory(require('../ApiClient'), require('../model/LyricsRequest'));
  } else {
    // Browser globals (root is window)
    if (!root.LyricsGeneration) {
      root.LyricsGeneration = {};
    }
    root.LyricsGeneration.LyricsApi = factory(root.LyricsGeneration.ApiClient, root.LyricsGeneration.LyricsRequest);
  }
}(this, function(ApiClient, LyricsRequest) {
  'use strict';

  /**
   * Lyrics service.
   * @module api/LyricsApi
   * @version 1.0.0
   */

  /**
   * Constructs a new LyricsApi. 
   * @alias module:api/LyricsApi
   * @class
   * @param {module:ApiClient} [apiClient] Optional API client implementation to use,
   * default to {@link module:ApiClient#instance} if unspecified.
   */
  var exports = function(apiClient) {
    this.apiClient = apiClient || ApiClient.instance;


    /**
     * Callback function to receive the result of the generateLyrics operation.
     * @callback module:api/LyricsApi~generateLyricsCallback
     * @param {String} error Error message, if any.
     * @param {'String'} data The data returned by the service call.
     * @param {String} response The complete HTTP response.
     */

    /**
     * Request lyric generation.
     * Request lyric generation.
     * @param {module:model/LyricsRequest} body LyricsRequest object that defines parameters of the lyrics desired.
     * @param {module:api/LyricsApi~generateLyricsCallback} callback The callback function, accepting three arguments: error, data, response
     * data is of type: {@link 'String'}
     */
    this.generateLyrics = function(body, callback) {
      var postBody = body;

      // verify the required parameter 'body' is set
      if (body === undefined || body === null) {
        throw new Error("Missing the required parameter 'body' when calling generateLyrics");
      }


      var pathParams = {
      };
      var queryParams = {
      };
      var collectionQueryParams = {
      };
      var headerParams = {
      };
      var formParams = {
      };

      var authNames = [];
      var contentTypes = ['application/json'];
      var accepts = ['application/json'];
      var returnType = 'String';

      return this.apiClient.callApi(
        '/lyrics', 'POST',
        pathParams, queryParams, collectionQueryParams, headerParams, formParams, postBody,
        authNames, contentTypes, accepts, returnType, callback
      );
    }
  };

  return exports;
}));
