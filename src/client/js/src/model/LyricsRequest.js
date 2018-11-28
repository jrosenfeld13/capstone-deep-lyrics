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
    define(['ApiClient'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    module.exports = factory(require('../ApiClient'));
  } else {
    // Browser globals (root is window)
    if (!root.LyricsGeneration) {
      root.LyricsGeneration = {};
    }
    root.LyricsGeneration.LyricsRequest = factory(root.LyricsGeneration.ApiClient);
  }
}(this, function(ApiClient) {
  'use strict';




  /**
   * The LyricsRequest model module.
   * @module model/LyricsRequest
   * @version 1.0.0
   */

  /**
   * Constructs a new <code>LyricsRequest</code>.
   * @alias module:model/LyricsRequest
   * @class
   */
  var exports = function() {
    var _this = this;








  };

  /**
   * Constructs a <code>LyricsRequest</code> from a plain JavaScript object, optionally creating a new instance.
   * Copies all relevant properties from <code>data</code> to <code>obj</code> if supplied or a new instance if not.
   * @param {Object} data The plain JavaScript object bearing properties of interest.
   * @param {module:model/LyricsRequest} obj Optional instance to populate.
   * @return {module:model/LyricsRequest} The populated <code>LyricsRequest</code> instance.
   */
  exports.constructFromObject = function(data, obj) {
    if (data) {
      obj = obj || new exports();

      if (data.hasOwnProperty('mode')) {
        obj['mode'] = ApiClient.convertToType(data['mode'], 'Number');
      }
      if (data.hasOwnProperty('danceability')) {
        obj['danceability'] = ApiClient.convertToType(data['danceability'], 'Number');
      }
      if (data.hasOwnProperty('energy')) {
        obj['energy'] = ApiClient.convertToType(data['energy'], 'Number');
      }
      if (data.hasOwnProperty('duration_ms')) {
        obj['duration_ms'] = ApiClient.convertToType(data['duration_ms'], 'Number');
      }
      if (data.hasOwnProperty('tempo')) {
        obj['tempo'] = ApiClient.convertToType(data['tempo'], 'Number');
      }
      if (data.hasOwnProperty('title')) {
        obj['title'] = ApiClient.convertToType(data['title'], 'String');
      }
      if (data.hasOwnProperty('genres')) {
        obj['genres'] = ApiClient.convertToType(data['genres'], ['String']);
      }
    }
    return obj;
  }

  /**
   * Song mode. 0 if minor, 1 if major.
   * @member {Number} mode
   */
  exports.prototype['mode'] = undefined;
  /**
   * @member {Number} danceability
   */
  exports.prototype['danceability'] = undefined;
  /**
   * @member {Number} energy
   */
  exports.prototype['energy'] = undefined;
  /**
   * @member {Number} duration_ms
   */
  exports.prototype['duration_ms'] = undefined;
  /**
   * @member {Number} tempo
   */
  exports.prototype['tempo'] = undefined;
  /**
   * Song title.
   * @member {String} title
   */
  exports.prototype['title'] = undefined;
  /**
   * comma-separated list of genres
   * @member {Array.<String>} genres
   */
  exports.prototype['genres'] = undefined;



  return exports;
}));


