import json
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import praw
from dotenv import load_dotenv
import os
import re
import logging
from pathlib import Path
import requests
from urllib.parse import urlparse

load_dotenv()  # Load environment variables from .env file

# logger
def setup_logger(log_folder):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_folder / f'data_gathering_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

log_folder = Path('./logs')
log_folder.mkdir(exist_ok=True)
logger = setup_logger(log_folder=log_folder)

class DataSource(ABC):
    @abstractmethod
    def search(self, keyword, limit, time_filter):
        pass

    @abstractmethod
    def get_content_and_comments(self, article_id, comment_limit):
        pass

    def limit_comments(self, comments, limit):
        return comments[:limit]

class RedditSource(DataSource):
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('CLIENT_ID'),
            client_secret=os.getenv('CLIENT_SECRET'),
            user_agent=os.getenv('USER_AGENT')
        )

    def search(self, keyword, limit, time_filter='all'):
        search_results = []
        for submission in self.reddit.subreddit('all').search(keyword, limit=limit, time_filter=time_filter):
            search_results.append({
                'title': submission.title,
                'href': submission.url,
                'published': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
            })
        
        logger.info(f"Found {len(search_results)} results for keyword: {keyword}")
        return search_results

    def get_content_and_comments(self, url, comment_limit):
        submission_id = re.search(r'comments/([a-zA-Z0-9]+)', url)
        if not submission_id:
            logger.warning(f"Could not extract submission ID from URL: {url}")
            return None, None

        submission_id = submission_id.group(1)
        try:
            submission = self.reddit.submission(id=submission_id)
            
            content = {
                'id': submission.id,
                'timestamp': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'title': submission.title,
                'url': submission.url,
                'body': submission.selftext
            }

            submission.comments.replace_more(limit=0)
            comments = []
            for comment in submission.comments.list()[:comment_limit]:
                comments.append({
                    'timestamp': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'body': comment.body
                })

            logger.info(f"Successfully retrieved content and {len(comments)} comments for submission: {submission_id}")
            return content, comments
        except Exception as e:
            logger.error(f"Error fetching submission {submission_id}: {str(e)}")
            return None, None

# Guardian-specific implementation
class GuardianSource(DataSource):
    def __init__(self):
        self.api_key = os.getenv('GUARDIAN_API_KEY')
        self.base_url = 'https://content.guardianapis.com'
        self.discussion_url = 'https://discussion.guardianapis.com/discussion-api'

    def search(self, keyword, limit, time_filter):
        from_date = self._get_from_date(time_filter)
        params = {
            'api-key': self.api_key,
            'q': keyword,
            'page-size': limit,
            'from-date': from_date,
            'show-fields': 'all',
            'show-tags': 'all',
            'show-elements': 'all'
        }
        response = requests.get(f"{self.base_url}/search", params=params)
        data = response.json()
        
        if 'response' not in data or 'results' not in data['response']:
            logger.warning(f"No search results found for keyword: {keyword}")
            return []
        
        search_results = []
        for result in data['response']['results']:
            search_results.append({
                'title': result['webTitle'],
                'href': result['webUrl'],
                'published': result['webPublicationDate']
            })
        
        logger.info(f"Found {len(search_results)} Guardian results for keyword: {keyword}")
        return search_results

    def _get_from_date(self, time_filter):
        now = datetime.now()
        if time_filter == 'day':
            return (now - timedelta(days=1)).strftime('%Y-%m-%d')
        elif time_filter == 'week':
            return (now - timedelta(weeks=1)).strftime('%Y-%m-%d')
        elif time_filter == 'month':
            return (now - timedelta(days=30)).strftime('%Y-%m-%d')
        elif time_filter == 'year':
            return (now - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            return None

    def get_content_and_comments(self, url, comment_limit):
        article_id = self._extract_article_id(url)
        content = self._get_content(article_id)
        comments = self._get_comments(article_id, comment_limit)
        
        if content:
            logger.info(f"Retrieved content for Guardian article: {article_id}")
        else:
            logger.warning(f"Failed to retrieve content for Guardian article: {article_id}")
        
        logger.info(f"Retrieved {len(comments)} comments for Guardian article: {article_id}")
        
        return content, comments

    def _extract_article_id(self, url):
        path = urlparse(url).path
        return path.strip('/')

    def _get_content(self, article_id):
        try:
            api_url = f"{self.base_url}/{article_id}"
            params = {
                'api-key': self.api_key,
                'show-fields': 'all',
                'show-tags': 'all',
                'show-elements': 'all'
            }
            response = requests.get(api_url, params=params)
            data = response.json()
            
            if 'response' not in data or 'content' not in data['response']:
                return None
            
            content = data['response']['content']
            
            return {
                'id': content['id'],
                'timestamp': content['webPublicationDate'],
                'title': content['webTitle'],
                'url': content['webUrl'],
                'body': content.get('fields', {}).get('body', '')
            }
        except Exception as e:
            logger.error(f"Error fetching content for article {article_id}: {str(e)}")
            return None

    def _get_comments(self, article_id, comment_limit):
        try:
            params = {
                'api-key': self.api_key,
                'discussionKey': article_id,
                'pageSize': comment_limit,
                'orderBy': 'oldest'
            }
            response = requests.get(f"{self.discussion_url}/discussion/{article_id}", params=params)
            data = response.json()
            
            comments = []
            for comment in data.get('discussion', {}).get('comments', []):
                comments.append({
                    'timestamp': comment['isoDateTime'],
                    'body': comment['body']
                })
            
            return comments
        except Exception as e:
            logger.error(f"Error fetching comments for article {article_id}: {str(e)}")
            return []

# LemmySource Class
class LemmySource(DataSource):
    def __init__(self):
        self.base_url = 'https://lemm.ee/api/v3'
        self.username = os.getenv('LEMMY_USERNAME')
        self.password = os.getenv('LEMMY_PASSWORD')
        self.jwt_token = self._login()

    def _login(self):
        login_url = f"{self.base_url}/user/login"
        login_data = {
            "username_or_email": self.username,
            "password": self.password
        }
        response = requests.post(login_url, json=login_data)
        if response.status_code == 200:
            return response.json()['jwt']
        else:
            logger.error(f"Failed to login to Lemm.ee: {response.text}")
            return None

    def search(self, keyword, limit, time_filter):
        search_url = f"{self.base_url}/search"
        params = {
            'q': keyword,
            'limit': limit,
            'sort': 'TopAll',  # Adjust sort as needed
            'type_': 'Posts'
        }
        headers = {
            'Authorization': f'Bearer {self.jwt_token}'
        }
        response = requests.get(search_url, params=params, headers=headers)
        data = response.json()
        
        if 'posts' not in data:
            logger.warning(f"No search results found for keyword: {keyword}")
            return []
        
        search_results = []
        for post in data['posts']:
            # Check if the keyword is in the post title or body
            if keyword.lower() in post['post']['name'].lower() or keyword.lower() in post['post']['body'].lower():
                search_results.append({
                    'title': post['post']['name'],
                    'href': f"https://lemm.ee/post/{post['post']['id']}",
                    'published': post['post']['published']
                })
        
        logger.info(f"Found {len(search_results)} Lemm.ee results for keyword: {keyword}")
        return search_results

    def get_content_and_comments(self, url, comment_limit):
        post_id = url.split('/')[-1]
        content_url = f"{self.base_url}/post"
        comments_url = f"{self.base_url}/comment/list"
        params = {
            'id': post_id,
        }
        headers = {
            'Authorization': f'Bearer {self.jwt_token}'
        }
        
        # Get content
        try:
            response = requests.get(content_url, params=params, headers=headers)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            post_data = response.json()
            if 'post_view' not in post_data:
                logger.warning(f"Failed to retrieve content for Lemm.ee post: {post_id}")
                return None, None
            
            post = post_data['post_view']['post']
            content = {
                'id': post['id'],
                'timestamp': post['published'],
                'title': post.get('name', 'No title'),
                'url': url,
                'body': post.get('body', 'No body content')
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching content for Lemm.ee post {post_id}: {str(e)}")
            return None, None
        except KeyError as e:
            logger.error(f"Unexpected response structure for Lemm.ee post {post_id}: {str(e)}")
            return None, None
        
        # Get comments
        try:
            comment_params = {
                'post_id': post_id,
                'limit': comment_limit,
                'sort': 'Old'  # Adjust sort as needed
            }
            response = requests.get(comments_url, params=comment_params, headers=headers)
            response.raise_for_status()
            comments_data = response.json()
            comments = []
            for comment in comments_data.get('comments', []):
                comments.append({
                    'timestamp': comment['comment']['published'],
                    'body': comment['comment'].get('content', 'No comment content')
                })
            
            comments = self.limit_comments(comments, comment_limit)
            
            logger.info(f"Successfully retrieved content and {len(comments)} comments for Lemm.ee post: {post_id}")
            return content, comments
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching comments for Lemm.ee post {post_id}: {str(e)}")
            return content, []
        except KeyError as e:
            logger.error(f"Unexpected response structure for comments of Lemm.ee post {post_id}: {str(e)}")
            return content, []

# Main function to handle all sources
def gather_data(sources, keywords, search_limit, comment_limit, time_filter):
    final_data = {}
    all_results = []

    for source in sources:
        source_name = source.__class__.__name__  # Get the class name as the source identifier
        for keyword in keywords:
            results = source.search(keyword, search_limit, time_filter)
            
            for result in results:
                result['source'] = source_name
            all_results.extend(results)

            for result in results:
                try:
                    content, comments = source.get_content_and_comments(result['href'], comment_limit)
                    if content or comments:
                        content_timestamp = content['timestamp'] if content else "Unknown"
                        content_title = content['title'] if content else result['title']
                        
                        comments_data = {comment['timestamp']: comment['body'] for comment in comments} if comments else {}

                        final_data[f"{content_timestamp} : {content_title}"] = {
                            'url': result['href'],
                            'source': source_name,
                            'content': content,
                            'comments': comments_data
                        }
                        
                        logger.info(f"Retrieved {'content and ' if content else ''}{len(comments)} comments for article: {content_title} from {source_name}")
                    else:
                        logger.warning(f"No content or comments found for URL: {result['href']} from {source_name}")
                except KeyError as e:
                    logger.error(f"KeyError processing URL {result['href']} from {source_name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing URL {result['href']} from {source_name}: {str(e)}")

    with open('./data/search_output.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    with open('./data/output_content_and_comments.json', 'w') as f:
        json.dump(final_data, f, indent=4)

    logger.info(f"Processed {len(all_results)} search results")
    logger.info(f"Extracted content and comments for {len(final_data)} items")

if __name__ == "__main__":
    sources = [RedditSource(), LemmySource()]
    keywords = ["government employee"]
    search_limit = 200
    comment_limit = 20
    time_filter = 'month'  # Options: 'hour', 'day', 'week', 'month', 'year', 'all'

    logger.info("Starting data gathering process")
    gather_data(sources, keywords, search_limit, comment_limit, time_filter)
    logger.info("Data gathering process completed")