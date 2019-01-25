#!/usr/bin/env node

/* eslint-disable id-length, no-console, no-process-env, no-sync, no-process-exit */
const fs = require('fs')
const { log } = console
const XmlStream = require('xml-stream')
const parseDate = require('./lib/parseDate')
const parseBody = require('./lib/parseBody')
const slugify = require('slugify')
function generateAuthorId (id) {
  return `author-${id}`
}

function generateCategoryId (id) {
  return `category-${id}`
}

function readFile (path = '') {
  if (!path) {
    return console.error('You need to set path')
  }
  return fs.createReadStream(path)
}

async function buildJSONfromStream (stream) {
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
     * Get the categories
     */
    const categories = []
    xml.on('endElement: category', wpCategory => {
      const { nicename } = wpCategory.$
      const category = {
        _type: 'category',
        _id: generateCategoryId(nicename),
        title: nicename
      }
      categories.push(category)
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
          current: slugify(author['wp:author_login'], { lower: true })
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
      const { title, category, link: permalink, description } = item
      if (item['wp:post_type'] != 'post' && item['wp:post_type'] != 'page') { return }
      const post = {
        _type: 'post',
        title,
        slug: {
          current: slugify(title, { lower: true })
        },
        categories: [
          {
            _type: 'reference',
            _ref: generateCategoryId(category.$.nicename)
          }
        ],
        description,
        body: parseBody(item['content:encoded']),
        publishedAt: parseDate(item)
        /* author: {
          _type: 'reference',
          _ref: users.find(user => user.slug.current === item['dc:creator'])._id
        },
        */
      }
      posts.push(post)
    })

    // there seems to be a bug where errors is not caught
    xml.on('error', err => {
      throw new Error(err)
    })

    xml.on('end', () => {
      const output = [
        /* meta, */
        ...users,
        ...posts,
        ...categories
      ]

      return res(output)
    })
  })
}

async function main () {
  const filename = './exampleData/alexoglou.xml'
  const stream = await readFile(filename)
  const output = await buildJSONfromStream(stream)
  output.forEach(doc => log(JSON.stringify(doc, null, 0)))
}

main()
