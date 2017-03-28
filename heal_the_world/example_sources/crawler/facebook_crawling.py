import json; import csv
import datetime; import time; import urllib2

def testFacebookPageData(page_id, access_token):
    base = "https://graph.facebook.com/v2.4"
    node = "/" + page_id
    parameters = "/?access_token=%s" % access_token
    url = base + node + parameters

    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    data = json.loads(response.read())

    print json.dumps(data, indent=4, sort_keys=True)

def request_until_succeed(url):
    req = urllib2.Request(url)
    success = False
    while success is False:
        try:
            response = urllib2.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception, e:
            print e
            time.sleep(5)

            print "Error for URL %s: %s" % (url, datetime.datetime.now())

    return response.read()

def testFacebookPageFeedData(page_id, access_token):
    base = "https://graph.facebook.com/v2.4"
    node = "/" + page_id + "/feed"
    parameters = "/?access_token=%s" % access_token
    url = base + node + parameters

    data = json.loads(request_until_succeed(url))

    print json.dumps(data, indent=4, sort_keys=True)


def getFacebookPageFeedData(page_id, access_token, num_statuses):
    base = "https://graph.facebook.com/v2.4"
    node = "/" + page_id + "/feed"
    parameters = "/?fields=message,link,created_time,type,name,id,likes.limit(1).summary(true),comments.limit(1).summary(true),shares&limit=%s&access_token=%s" % (num_statuses, access_token)
    url = base + node + parameters

    data = json.loads(request_until_succeed(url))

    return data

def processFacebookPageFeedStatus(status):
    status_id = status['id']
    status_message = '' if 'message' not in status.keys() else status['message'].encode('utf-8')
    link_name = '' if 'name' not in status.keys() else status['name'].encode('utf-8')
    status_type = status['type']
    status_link = '' if 'link' not in status.keys() else status['link']

    status_published = datetime.datetime.strptime(status['created_time'], '%Y-%m-%dT%H:%M:%S+0000')
    status_published = status_published + datetime.timedelta(hours=-5)
    status_published = status_published.strftime('%Y-%m-%d %H:%M:%S')

    num_likes = 0 if 'likes' not in status.keys() else status['likes']['summary']['total_count']
    num_comments = 0 if 'comments' not in status.keys() else status['comments']['summary']['total_count']
    num_shares = 0 if 'shares' not in status.keys() else status['shares']['count']

    return (status_id, status_message, link_name, status_type, status_link, status_published, num_likes, num_comments, num_shares)

def scrapeFacebookPageFeedStatus(page_id, access_token):
    with open('%s_facebook_statuses.csv' % page_id, 'wb') as file:
        w = csv.writer(file)
        w.writerow(["status_id", "status_message", "link_name", "status_type", "status_link",
                    "status_published", "num_likes", "num_comments", "num_shares"])

        has_next_page = True
        num_processed = 0
        scrape_starttime = datetime.datetime.now()

        print "Scraping %s Facebook Page: %s\n" % (page_id, scrape_starttime)

        statuses = getFacebookPageFeedData(page_id, access_token, 100)

        while has_next_page:
            for status in statuses['data']:
                w.writerow(processFacebookPageFeedStatus(status))

                num_processed += 1
                if num_processed % 1000 == 0:
                    print "%s Statuses Processed: %s" % (num_processed, datetime.datetime.now())

            if 'paging' in statuses.keys():
                statuses = json.loads(request_until_succeed(statuses['paging']['next']))
            else:
                has_next_page = False

        print "\nDone!\n%s Statuses Processed in %s" % (num_processed, datetime.datetime.now() - scrape_starttime)

if __name__ == "__main__":
   app_id = "405325483132744"
   app_secret = "e1674998dffc6f32a750fa0c725be7c9"

   access_token = app_id + "|" + app_secret

   page_id = 'nytimes'

   scrapeFacebookPageFeedStatus(page_id, access_token)
