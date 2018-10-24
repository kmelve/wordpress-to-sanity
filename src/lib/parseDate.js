module.exports = item => {
  let date = item['wp:post_date_gmt'];
  if (date == undefined || date == "0000-00-00 00:00:00") {
    date = item['wp:post_date'];
    if (date == undefined) {
      date = '1970-01-01 00:00:00'
    }
  }

  date = date.match(/(\d{4})-(\d+)-(\d+) (\d+):(\d+):(\d+)/);
  date = date.map((e) => {
    return parseInt(e, 10);
  });
  let pubDate = new Date(Date.UTC(date[1], date[2] - 1, date[3], date[4], date[5], date[6], 0));


  if (item.pubDate.match("-0001") === null) {
    pubDate = new Date(item.pubDate);
  }
  return pubDate
}