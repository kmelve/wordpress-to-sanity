const sanitizeHTML = require('sanitize-html')

module.exports = html => sanitizeHTML(html, {
  allowedTags: [
    'a',
    'audio',
    'b',
    'blockquote',
    'br',
    'caption',
    'code',
    'div',
    'em',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'hr',
    'i',
    'img',
    'li',
    'nl',
    'ol',
    'p',
    'pre',
    'strike',
    'strong',
    'table',
    'tbody',
    'td',
    'th',
    'thead',
    'tr',
    'ul',
    'video'
  ]
})