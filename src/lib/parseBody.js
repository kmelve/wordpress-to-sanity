const {
  JSDOM
} = require('jsdom');
const Schema = require('@sanity/schema').default;
const blockTools = require('@sanity/block-tools').default;
const sanitizeHTML = require('sanitize-html')



const schema = Schema.compile({
  name: 'default',
  types: [{
    type: 'object',
    name: 'mock',
    fields: [{
      title: 'Body',
      name: 'body',
      type: 'array',
      of: [{
        type: 'block'
      }, {
        type: 'image'
      }],
    }, ],
  }, ],
});

const blockContentType = schema
  .get('mock')
  .fields.find(field => field.name === 'body').type;

const extractImages = (el, next) => {
  if (
    el.tagName === 'P' &&
    el.childNodes.length === 1 &&
    el.childNodes[0].tagName === 'IMG'
  ) {
    return {
      _sanityAsset: `image@${el.childNodes[0]
        .getAttribute('src')
        .replace(/^\/\//, 'https://')}`,
    };
  }

  // Only convert block-level images, for now
  return undefined;
};

function htmlToBlocks(html, options) {
  if (!html) {
    return [];
  }
  const sanitizedHTML = sanitizeHTML(html, {
    allowedTags: ['h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'br', 'p', 'a', 'ul', 'ol',
      'nl', 'li', 'b', 'i', 'strong', 'em', 'strike', 'code', 'hr', 'div',
      'table', 'thead', 'caption', 'tbody', 'tr', 'th', 'td', 'pre', 'video', 'audio'
    ]
  })

  const blocks = blockTools.htmlToBlocks(sanitizedHTML, {
    rules: [{
      deserialize: extractImages
    }],
    blockContentType,
    parseHtml: htmlContent => new JSDOM(htmlContent).window.document,
  });

  return blocks;
};



module.exports = item => {
  const bodyHTML = item['content:encoded']
  const body = htmlToBlocks(bodyHTML)
  return body
}