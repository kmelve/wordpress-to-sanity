# Wordpress to Sanity

Forked from [wordpress-to-sanity](https://github.com/kmelve/wordpress-to-sanity)

This is primarily an example of how you can script a Wordpress to [Sanity.io](https://www.sanity.io) migration.
You most probably have to tweak and edit the code to fit to your need.

## Getting started

1. Clone this repo
2. Update `const filename` in `migrate.js` to the path of your wordpress export xml file
3. Run `node src/migrate.js` to log out the converted sanity documents in the terminal
4. Pipe the output to an import file by `node src/migrate.js > myImportfile.ndjson`
5. Copy the `myImportfile.ndjson` to your Sanity project folder.
6. Remember to create your Sanity schema for Post, Category, etc.
7. Import the file with `sanity dataset import myImportfile.ndjson` in your Sanity project folder

Mostly probably there is additional content that this script doesn't migrate, however, it should cover the most usual patterns, so that you can build it out for your specific use case.


- This script reads the wordpress export XML file
- This example is based on the blog template schema
- There's a bug in `xml-stream` where it doesn't seem to emit errors
- Debugging deserialization (`parseBody.js`) is easier with the script in `/test`
- Remember to add additional supported custom types to `/schemas/defaultSchema.js`
- The HTML is sanitized, but _most_ HTML tags are allowed (check `/lib/sanitizeHTML.js`)
- 