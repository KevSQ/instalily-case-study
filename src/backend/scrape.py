import scrapy
from scrapy.loader import ItemLoader
from itemloaders.processors import TakeFirst, MapCompose, Join
from w3lib.html import remove_tags

class PartSelectItem(scrapy.Item):
    part_name = scrapy.Field()
    part_number = scrapy.Field()
    manufacturer = scrapy.Field()
    price = scrapy.Field()
    availability = scrapy.Field()
    description = scrapy.Field()
    rating = scrapy.Field()
    review_count = scrapy.Field()
    reviews = scrapy.Field()

class PartSelectSpider(scrapy.Spider):
    name = 'partselect'
    start_urls = ['https://www.partselect.com/PS3579318-LG-4681EA2001T-Washing-Machine-Drain-Pump.htm?SourceCode=18#Instructions']

    def parse(self, response):
        l = ItemLoader(item=PartSelectItem(), response=response)

        l.add_css('part_name', 'h1::text', TakeFirst())
        l.add_css('part_number', '.mt-3.mb-2:contains("PartSelect Number") ::text', TakeFirst(), re=r'PS\d+')
        l.add_css('manufacturer', '.mb-2:contains("Manufactured by") ::text', TakeFirst(), re=r'LG|Kenmore')
        l.add_css('price', '.price::text', TakeFirst(), re=r'\d+\.\d+')
        l.add_css('availability', '.js-partAvailability::text', TakeFirst())
        l.add_css('description', '.pd__description p::text', Join())
        l.add_css('rating', '.rating__stars__upper::attr(style)', TakeFirst(), re=r'\d+')
        l.add_css('review_count', '.rating__count::text', TakeFirst(), re=r'\d+')

        reviews = []
        for review in response.css('.pd__cust-review__submitted-review'):
            review_data = {
                'rating': len(review.css('.rating__stars__upper')),
                'author': review.css('.bold::text').get(),
                'date': review.css('.pd__cust-review__submitted-review__header::text').get(),
                'content': review.css('.js-searchKeys::text').get(),
            }
            reviews.append(review_data)

        l.add_value('reviews', reviews)

        yield l.load_item()

class PartSelectPipeline:
    def process_item(self, item, spider):
        # Convert rating to a percentage
        if 'rating' in item:
            item['rating'] = f"{float(item['rating'])}%"
        return item

# Add this to your Scrapy settings
ITEM_PIPELINES = {
    'partselect_spider.PartSelectPipeline': 300,
}