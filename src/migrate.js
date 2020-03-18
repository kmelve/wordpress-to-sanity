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

function generateTagId(id) {
  return `tag-${id}`
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
     * Collate unique categories and tags
     */
    const categories = [];
    const tags = [];
    xml.on('endElement: category', wpCategory => {
      const { nicename, domain } = wpCategory.$

      // Unique categories only, no tags
      if (domain === 'category' && !categories.find(cat => cat.title === nicename)) {
        const category = {
          _type: 'category',
          _id: generateCategoryId(nicename),
          title: nicename
        }

        categories.push(category)
      }

      // Unique tags only, no categories
      if (domain === 'post_tag' && !tags.find(tag => tag.title === nicename)) {
        const tag = {
          _type: 'tag',
          _id: generateTagId(nicename),
          title: nicename
        }

        tags.push(tag)
      }
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
    xml.collect('category')
    xml.on('endElement: item', item => {
      const { title, category, description } = item
      if (item['wp:post_type'] != 'post' && item['wp:post_type'] != 'page') { return }
      const post = {
        _type: 'post',
        _id: `post-${item['wp:post_id']}`,
        title,
        slug: {
          current: item['wp:post_name']
        },
        description,
        body: parseBody(item['content:encoded']),
        publishedAt: parseDate(item)
        /* author: {
          _type: 'reference',
          _ref: users.find(user => user.slug.current === item['dc:creator'])._id
        },
        */
      }

      const postCategories = [];
      const postTags = [];

      // Add categories and tags as arrays of references
      if (category.length > 0) {
        category.forEach(cat => {
          if (cat.$.domain === 'category') {
            postCategories.push({
              _type: 'category',
              _ref: generateCategoryId(cat.$.nicename)
            })
          }

          if (cat.$.domain === 'post_tag') {
            postTags.push({
              _type: 'tag',
              _ref: generateTagId(cat.$.nicename)
            })
          }
        })
      }

      post.categories = postCategories;
      post.tags = postTags;

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
        ...categories,
        ...tags
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
