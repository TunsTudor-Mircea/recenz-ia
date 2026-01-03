"""
Web scraper service for Romanian e-commerce sites.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
import time
from datetime import datetime
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


class ReviewData:
    """Data class for scraped review."""
    def __init__(self, product_name: str, review_text: str, rating: int, review_date: Optional[datetime] = None, review_title: Optional[str] = None, **metadata):
        self.product_name = product_name
        self.review_title = review_title
        self.review_text = review_text
        self.rating = rating
        self.review_date = review_date
        self.metadata = metadata


class BaseScraper:
    """Base scraper class."""

    def __init__(self, url: str):
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_page(self) -> Optional[str]:
        """Fetch page content."""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching page {self.url}: {e}")
            return None

    def scrape(self) -> List[ReviewData]:
        """Scrape reviews from the page. Must be implemented by subclasses."""
        raise NotImplementedError


class SeleniumScraper(BaseScraper):
    """Base scraper with Selenium support for JavaScript-heavy pages."""

    def __init__(self, url: str):
        super().__init__(url)
        self.driver = None

    def setup_driver(self, retries: int = 3) -> webdriver.Chrome:
        """
        Setup headless Chrome driver with retry logic.

        Args:
            retries: Number of retries for driver setup

        Returns:
            Configured Chrome WebDriver instance

        Raises:
            WebDriverException: If driver setup fails after all retries
        """
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')

        # Disable images and CSS for faster loading
        prefs = {
            'profile.managed_default_content_settings.images': 2,
            'profile.managed_default_content_settings.stylesheets': 2
        }
        chrome_options.add_experimental_option('prefs', prefs)

        last_exception = None
        for attempt in range(retries):
            try:
                driver = webdriver.Chrome(options=chrome_options)
                logger.info("Chrome driver initialized successfully")
                return driver
            except WebDriverException as e:
                last_exception = e
                logger.warning(f"Failed to setup Chrome driver (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to setup Chrome driver after {retries} attempts")
        raise last_exception

    def fetch_page_with_js(self, wait_for_selector: Optional[str] = None, wait_time: int = 10) -> Optional[str]:
        """
        Fetch page content with JavaScript rendering.

        Args:
            wait_for_selector: CSS selector to wait for before returning content
            wait_time: Maximum time to wait in seconds

        Returns:
            Page HTML after JavaScript execution
        """
        try:
            self.driver = self.setup_driver()
            self.driver.get(self.url)

            if wait_for_selector:
                try:
                    WebDriverWait(self.driver, wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                    )
                except TimeoutException:
                    logger.warning(f"Timeout waiting for selector: {wait_for_selector}")
            else:
                # Default wait for page load
                time.sleep(3)

            # Scroll to load lazy-loaded content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            html = self.driver.page_source
            return html

        except Exception as e:
            logger.error(f"Error fetching page with Selenium {self.url}: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None


class EmagScraper(SeleniumScraper):
    """Scraper for eMAG.ro product pages using Selenium for dynamic content."""

    def extract_product_name(self, soup: BeautifulSoup) -> str:
        """Extract product name from eMAG page."""
        # Try multiple selectors
        selectors = [
            'h1.page-title',
            'h1[itemprop="name"]',
            '.page-title',
            'h1.product-title',
            'h1'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return "Unknown Product"

    def extract_rating_from_stars(self, review_elem) -> int:
        """Extract rating from star elements."""
        # eMAG uses class names like "rated-5", "rated-4", etc.
        star_rating_elem = review_elem.select_one('.star-rating')
        if star_rating_elem:
            # Look for rated-X class
            classes = star_rating_elem.get('class', [])
            for cls in classes:
                if cls.startswith('rated-'):
                    try:
                        rating = int(cls.split('-')[1])
                        return min(5, max(1, rating))
                    except (ValueError, IndexError):
                        pass

        # Fallback: count filled stars
        filled_stars = review_elem.select('.star-filled, .rating-stars .fa-star:not(.fa-star-o)')
        if filled_stars:
            return min(5, max(1, len(filled_stars)))

        # Try data attributes
        rating_elem = review_elem.select_one('[data-rating]')
        if rating_elem and rating_elem.get('data-rating'):
            try:
                rating = int(float(rating_elem.get('data-rating')))
                return min(5, max(1, rating))
            except (ValueError, TypeError):
                pass

        # Try to find rating in text like "5 out of 5" or "5/5"
        rating_text = review_elem.get_text()
        rating_match = re.search(r'(\d)\s*(?:out\s*of\s*5|/\s*5|din\s*5)', rating_text)
        if rating_match:
            return int(rating_match.group(1))

        # Default to 5 if we find the review but can't determine rating
        return 5

    def extract_review_date(self, review_elem) -> Optional[datetime]:
        """Extract review date from eMAG review element."""
        try:
            # Look for date in review meta area or user meta
            # Format is typically "18 Dec 2025" or "18 decembrie 2025"
            date_patterns = [
                r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
                r'(\d{1,2})\s+(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})'
            ]

            # Romanian month mapping
            month_mapping = {
                'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4,
                'mai': 5, 'iunie': 6, 'iulie': 7, 'august': 8,
                'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12,
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }

            # Get text from the entire review element
            review_text = review_elem.get_text()

            for pattern in date_patterns:
                match = re.search(pattern, review_text)
                if match:
                    day = int(match.group(1))
                    month_str = match.group(2)
                    year = int(match.group(3))
                    month = month_mapping.get(month_str, 1)

                    try:
                        review_date = datetime(year, month, day)
                        return review_date
                    except ValueError:
                        continue

            return None

        except Exception as e:
            logger.debug(f"Error extracting review date: {e}")
            return None

    def scrape(self) -> List[ReviewData]:
        """Scrape reviews from eMAG product page using Selenium."""
        # Use Selenium to load JavaScript content
        # Wait for review elements to load
        html = self.fetch_page_with_js(
            wait_for_selector='.card-body, .review-item, [class*="review"]',
            wait_time=15
        )

        if not html:
            logger.warning(f"Failed to fetch page with Selenium: {self.url}")
            return []

        soup = BeautifulSoup(html, 'html.parser')
        product_name = self.extract_product_name(soup)
        reviews = []

        # Debug: Log all classes that contain 'review' to help identify correct selectors
        all_review_classes = set()
        for elem in soup.find_all(class_=True):
            for class_name in elem.get('class', []):
                if 'review' in class_name.lower() or 'comment' in class_name.lower() or 'rating' in class_name.lower():
                    all_review_classes.add(class_name)
        logger.info(f"Found classes containing 'review/comment/rating': {sorted(all_review_classes)}")

        # eMAG review selectors (updated based on actual page structure from debug)
        # Try multiple possible selectors
        review_selectors = [
            '.product-review-item',  # Most specific eMAG review container
            '.js-review-item',  # JavaScript-enabled review item
            '[class*="product-review-item"]',
            '[class*="js-review-item"]',
            '.card-body .review-comment-item',  # Legacy structure
            '.pdp-reviews-item',
            '.review-item',
            '.customer-review',
            '[itemprop="review"]',
            '[class*="ReviewItem"]',
            '.em-review-item'
        ]

        review_elements = []
        for selector in review_selectors:
            review_elements = soup.select(selector)
            if review_elements:
                logger.info(f"Found {len(review_elements)} review elements with selector: {selector}")
                break

        # If no structured reviews found, look for any card-body content
        if not review_elements:
            logger.info("No structured reviews found, trying card-body sections")
            card_bodies = soup.select('.card-body')
            for card in card_bodies:
                # Check if this card contains review-like content
                text = card.get_text(strip=True)
                if len(text) > 50 and not any(skip in text.lower() for skip in ['caracteristici', 'specificatii', 'detalii']):
                    review_elements.append(card)

        logger.info(f"Processing {len(review_elements)} potential review elements")

        for idx, review_elem in enumerate(review_elements[:20]):  # Limit to 20 reviews
            # Extract review title (optional)
            title_elem = review_elem.select_one('.product-review-title, h3')
            review_title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract review body - MUST use the specific review body container
            # This is the key element that contains the actual review text
            text_elem = review_elem.select_one('.js-review-body, .review-body-container, .mb-2.js-review-body')

            if not text_elem:
                # Fallback: try other body selectors
                text_elem = review_elem.select_one(
                    '.product-review-body, .review-text, .review-body, .review-content, '
                    '[itemprop="reviewBody"], .customer-review-text'
                )

            if not text_elem:
                # Last resort: look for divs with substantial text, excluding metadata
                logger.debug(f"Review {idx}: No review body found with standard selectors, trying fallback")
                paragraphs = review_elem.select('div, p')
                for p in paragraphs:
                    # Skip if this is an author, title, or meta element
                    if p.get('class'):
                        classes = ' '.join(p.get('class', []))
                        # Skip metadata elements
                        if any(skip in classes for skip in ['product-review-author', 'product-review-user',
                                                             'product-review-head', 'product-review-title',
                                                             'review-meta', 'star-rating', 'product-review-user-avatar']):
                            continue

                    text = p.get_text(strip=True)
                    # Must be substantial text
                    if len(text) >= 20 and not text.startswith(('Detii', 'Review', 'stele', 'Adauga', 'clienti', 'Spune', 'Ai')):
                        text_elem = p
                        logger.debug(f"Review {idx}: Found fallback text: {text[:50]}...")
                        break

            if not text_elem:
                logger.debug(f"Review {idx}: Skipped - no review body found")
                continue

            review_body = text_elem.get_text(strip=True)

            # Clean up review body - remove eMAG metadata text
            review_body = re.sub(r'Review (cumparator|utilizator) eMAG', '', review_body, flags=re.IGNORECASE).strip()

            # Keep title and body separate - don't combine them
            # The title will go into review_title field, body into review_text
            review_text = review_body

            logger.debug(f"Review {idx}: Title='{review_title[:50] if review_title else 'N/A'}', Body length={len(review_body)}")

            # Filter out UI elements and metadata
            # Skip if it's clearly not a review
            # If there's only a title but no body, use the title as the text
            if not review_text and review_title:
                review_text = review_title
                review_title = None  # Clear title since we're using it as body

            if len(review_text) < 20:
                logger.debug(f"Review {idx}: Filtered - text too short ({len(review_text)} chars)")
                continue
            if review_text.startswith(('Detii', 'Adauga', 'Review', 'stele', 'clienti eMAG', 'Spune', 'Ai cumparat', 'Scrie')):
                logger.debug(f"Review {idx}: Filtered - starts with UI text: {review_text[:50]}")
                continue
            if 'review-uri' in review_text or 'parerea acordand' in review_text or 'recomanda produsul' in review_text:
                logger.debug(f"Review {idx}: Filtered - contains meta text")
                continue
            if re.match(r'^\d+%', review_text) or re.match(r'^\d+\s*stele?\(', review_text):
                logger.debug(f"Review {idx}: Filtered - stat/rating pattern")
                continue
            # Allow reviews with at least 2 words (more lenient for short but valid reviews)
            if review_text.count(' ') < 1:
                logger.debug(f"Review {idx}: Filtered - too few words (spaces: {review_text.count(' ')})")
                continue

            # Extract rating
            rating = self.extract_rating_from_stars(review_elem)

            # Extract review date - use today's date if extraction fails
            review_date = self.extract_review_date(review_elem)
            if not review_date:
                review_date = datetime.now()
                logger.debug(f"Review {idx}: Could not extract date, using current date")

            reviews.append(ReviewData(
                product_name=product_name,
                review_title=review_title if review_title else None,
                review_text=review_text,
                rating=rating,
                review_date=review_date,
                source='emag'
            ))

        logger.info(f"Scraped {len(reviews)} reviews from eMAG: {product_name}")
        return reviews


class CelScraper(BaseScraper):
    """Scraper for Cel.ro product pages."""

    def extract_product_name(self, soup: BeautifulSoup) -> str:
        """Extract product name from Cel.ro page."""
        selectors = [
            'h1.product_title',
            'h1[itemprop="name"]',
            '.product-name',
            'h1'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return "Unknown Product"

    def scrape(self) -> List[ReviewData]:
        """Scrape reviews from Cel.ro product page."""
        html = self.fetch_page()
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        product_name = self.extract_product_name(soup)
        reviews = []

        # Find review elements
        review_elements = soup.select('.review, .customer-review, [class*="review-item"]')

        for review_elem in review_elements[:20]:
            text_elem = review_elem.select_one('.review-text, .comment-text, p')
            if not text_elem:
                continue

            review_text = text_elem.get_text(strip=True)
            if len(review_text) < 10:
                continue

            # Extract rating (default to 3 if not found)
            rating = 3
            rating_elem = review_elem.select_one('[class*="rating"], [class*="star"]')
            if rating_elem:
                rating_text = rating_elem.get_text()
                numbers = re.findall(r'\d+', rating_text)
                if numbers:
                    rating = min(5, max(1, int(numbers[0])))

            reviews.append(ReviewData(
                product_name=product_name,
                review_text=review_text,
                rating=rating,
                review_date=datetime.now(),  # Use current date as fallback
                source='cel'
            ))

        logger.info(f"Scraped {len(reviews)} reviews from Cel.ro: {product_name}")
        return reviews


class GenericScraper(BaseScraper):
    """Generic scraper for unknown e-commerce sites."""

    def scrape(self) -> List[ReviewData]:
        """Scrape reviews using generic patterns."""
        html = self.fetch_page()
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # Try to find product name
        product_name = "Unknown Product"
        title_elem = soup.select_one('h1')
        if title_elem:
            product_name = title_elem.get_text(strip=True)

        reviews = []

        # Look for common review patterns
        review_keywords = ['review', 'comment', 'feedback', 'opinion', 'recenzie', 'comentariu']

        for keyword in review_keywords:
            review_elements = soup.find_all(class_=re.compile(keyword, re.I))

            for review_elem in review_elements[:20]:
                text = review_elem.get_text(strip=True)

                if len(text) < 10 or len(text) > 1000:
                    continue

                reviews.append(ReviewData(
                    product_name=product_name,
                    review_text=text,
                    rating=3,  # Default rating
                    review_date=datetime.now(),  # Use current date as fallback
                    source='generic'
                ))

        logger.info(f"Scraped {len(reviews)} reviews using generic scraper: {product_name}")
        return reviews


def get_scraper(url: str, site_type: Optional[str] = None) -> BaseScraper:
    """Get appropriate scraper based on URL or site type."""

    if site_type:
        site_type = site_type.lower()

    # Detect site type from URL if not provided
    if not site_type:
        if 'emag.ro' in url.lower():
            site_type = 'emag'
        elif 'cel.ro' in url.lower():
            site_type = 'cel'

    # Return appropriate scraper
    if site_type == 'emag':
        return EmagScraper(url)
    elif site_type == 'cel':
        return CelScraper(url)
    else:
        return GenericScraper(url)


def scrape_reviews(url: str, site_type: Optional[str] = None) -> List[ReviewData]:
    """
    Scrape reviews from a URL.

    Args:
        url: URL of the product page
        site_type: Optional site type (emag, cel, etc.)

    Returns:
        List of ReviewData objects
    """
    scraper = get_scraper(url, site_type)
    return scraper.scrape()
