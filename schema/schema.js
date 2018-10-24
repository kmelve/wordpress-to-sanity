const Schema = require('@sanity/schema').default

/* const { getImageAttachment, getFileAttachment } = require('./getAttachment')
const postSchema = require('./helpers/post')
const funkyTable = require('./helpers/funkyTable')
const randomKey = require('./helpers/randomKey')
const unhandledRejection = require('./helpers/unhandledRejection');
unhandledRejection()
 */
const blockContent = require('./blockContent')
const category = require('./category')
const post = require('./post')
const author = require('./author')

const schema = Schema.compile({
    name: 'default',
    types: [
        blockContent,
        category,
        post,
        author
    ]
  })