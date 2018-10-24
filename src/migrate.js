#!/usr/bin/env node

/* eslint-disable id-length, no-console, no-process-env, no-sync, no-process-exit */
const fs = require('fs')
const {log} = console
const XmlStream = require('xml-stream')
const parseDate = require('./lib/parseDate')
const parseBody = require('./lib/parseBody')

function generateAuthorId(id) {
  return `author-${id}`
}

function readFile(path = '') {
  if (!path) {
    return console.error('You need to set path')
  }
  return fs.createReadStream(path)
}

async function buildJSONfromStream(stream) {
  const xml = await new XmlStream(stream)

  return new Promise((res, rej) => {
    /**
     * Get some meta info
     */
    const meta = {}
    xml.on('text: wp:base_site_url', url => {
      meta.rootUrl = url.$text
    })

    /**
     * Get the users
     */
    const users = []
    xml.on('endElement: wp:author', author => {
      const user = {
        _type: 'author',
        _id: generateAuthorId(author['wp:author_id']),
        name: author['wp:author_display_name'],
        slug: {
          current: author['wp:author_login']
        },
        email: author['wp:author_email']
      }
      users.push(user)
    })

    /**
     * Get the posts
     */
    const posts = []
    xml.collect('wp:postmeta')
    xml.on('endElement: item', item => {
      const {title, link: permalink, description} = item
      if (item['wp:post_type'] != 'post' && item['wp:post_type'] != 'page') return
      const post = {
        _type: 'post',
        title,
        meta: {
          permalink,
          featuredImage: {
            _sanityAsset: `image@${meta.rootUrl}`,
            t: item['wp:postmeta']
          }
        },
        description,
        slug: {
          current: ''
        },
        author: {
          _type: 'reference',
          _ref: users.find(user => user.slug.current === item['dc:creator'])._id || 'author-unknown'
        },
        categories: [],
        publishedAt: parseDate(item),
        body: parseBody(item)
      }
      posts.push(post)
    })

    xml.on('end', () => {
      const output = {
        meta,
        users,
        posts
      }
      return res(output)
    })
  })
}

async function main() {
  const filename = 'PUT YOUR XML FILENAME HERE'
  const stream = await readFile(filename)
  const output = await buildJSONfromStream(stream)
  log(JSON.stringify(output, null, 2))
}

main()
