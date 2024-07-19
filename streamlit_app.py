import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import random
import time
from datetime import datetime, timedelta
import re
import requests
import json

# User agents list
user_agents = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
]

def make_header():
    return {'User-Agent': random.choice(user_agents)}

async def extract_by_article(url, semaphore):
    async with semaphore:
        async with aiohttp.ClientSession(headers=make_header()) as session:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        st.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                    data = await response.text()
                    soup = BeautifulSoup(data, "lxml")
                    
                    def get_text(element):
                        return element.text.strip() if element else 'N/A'

                    title = get_text(soup.find('h1', {'class': 'heading-title'}))
                    
                    abstract_div = soup.find('div', {'id': 'abstract'})
                    
                    background = results = conclusion = keywords = abstract = 'N/A'
                    
                    if abstract_div:
                        abstract_content = abstract_div.find('div', {'class': 'abstract-content selected'})
                        if abstract_content:
                            abstract = ' '.join([p.text.strip() for p in abstract_content.find_all('p')])
                            
                            for p in abstract_content.find_all('p'):
                                strong = p.find('strong', class_='sub-title')
                                if strong:
                                    section_title = strong.text.strip().lower()
                                    content = p.text.replace(strong.text, '').strip()
                                    
                                    if 'background' in section_title:
                                        background = content
                                    elif 'results' in section_title:
                                        results = content
                                    elif 'conclusion' in section_title:
                                        conclusion = content
                        
                        if background == 'N/A' and abstract != 'N/A':
                            background = abstract

                    keywords_p = soup.find('p', class_='keywords')
                    if keywords_p:
                        keywords = keywords_p.text.replace('Keywords:', '').strip()
                    else:
                        keyword_match = re.search(r'Keywords?:?\s*(.*?)(?:\.|$)', abstract, re.IGNORECASE | re.DOTALL)
                        if keyword_match:
                            keywords = keyword_match.group(1).strip()
                    
                    date_elem = soup.find('span', {'class': 'cit'}) or soup.find('time', {'class': 'citation-year'})
                    date = get_text(date_elem)
                    
                    journal_elem = soup.find('button', {'id': 'full-view-journal-trigger'}) or soup.find('span', {'class': 'journal-title'})
                    journal = get_text(journal_elem)
                    
                    doi_elem = soup.find('span', {'class': 'citation-doi'})
                    doi = get_text(doi_elem).replace('doi:', '').strip()

                    copyright_elem = soup.find('div', class_='copyright-section') or soup.find('p', class_='copyright')
                    copyright_text = get_text(copyright_elem)

                    affiliations = {}
                    affiliations_div = soup.find('div', {'class': 'affiliations'})
                    if affiliations_div:
                        for li in affiliations_div.find_all('li'):
                            sup = li.find('sup')
                            if sup:
                                aff_num = sup.text.strip()
                                aff_text = li.text.replace(aff_num, '').strip()
                                affiliations[aff_num] = aff_text

                    authors_div = soup.find('div', {'class': 'authors-list'})
                    author_affiliations = []
                    if authors_div:
                        for author in authors_div.find_all('span', {'class': 'authors-list-item'}):
                            name = author.find('a', {'class': 'full-name'})
                            if name:
                                author_name = name.text.strip()
                                author_aff_nums = [sup.text.strip() for sup in author.find_all('sup')]
                                author_affs = [affiliations.get(num, '') for num in author_aff_nums]
                                author_affiliations.append((author_name, '; '.join(author_affs)))

                    pmid_elem = soup.find('strong', string='PMID:')
                    pmid = pmid_elem.next_sibling.strip() if pmid_elem else 'N/A'

                    pub_type_elem = soup.find('span', {'class': 'publication-type'})
                    pub_type = get_text(pub_type_elem)

                    mesh_terms = []
                    mesh_div = soup.find('div', {'class': 'mesh-terms'})
                    if mesh_div:
                        mesh_terms = [term.text.strip() for term in mesh_div.find_all('li')]

                    return {
                        'url': url,
                        'title': title,
                        'authors': author_affiliations,
                        'abstract': abstract,
                        'background': background,
                        'results': results,
                        'conclusion': conclusion,
                        'keywords': keywords,
                        'date': date,
                        'journal': journal,
                        'doi': doi,
                        'copyright': copyright_text,
                        'pmid': pmid,
                        'publication_type': pub_type,
                        'mesh_terms': mesh_terms
                    }
            except asyncio.TimeoutError:
                st.warning(f"Timeout while fetching {url}")
                return None
            except Exception as e:
                st.warning(f"Error processing {url}: {str(e)}")
                return None

async def get_pmids(page, query, filters, session):
    base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
    params = f'term={query}&{filters}&page={page}'
    url = f'{base_url}?{params}'
    
    async with session.get(url) as response:
        data = await response.text()
        soup = BeautifulSoup(data, "lxml")
        pmids = soup.find('meta', {'name': 'log_displayeduids'})
        if pmids:
            return [f"{base_url}{pmid}" for pmid in pmids['content'].split(',')]
        return []

async def scrape_pubmed(query, filters, num_pages):
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    all_urls = []
    async with aiohttp.ClientSession(headers=make_header()) as session:
        for page in range(1, num_pages + 1):
            urls = await get_pmids(page, query, filters, session)
            all_urls.extend(urls)
            if len(urls) < 10:  # Less than 10 results on a page means it's the last page
                break
    
    results = []
    for url in all_urls:
        try:
            result = await extract_by_article(url, semaphore)
            if result:
                results.append(result)
        except Exception as e:
            st.warning(f"Error processing {url}: {str(e)}")
    
    return results

def parse_author_info(authors):
    parsed_authors = []
    for index, (author, affiliation) in enumerate(authors):
        name_parts = author.split()
        if len(name_parts) > 1:
            first_name = name_parts[0]
            last_name = ' '.join(name_parts[1:])
        else:
            first_name = author
            last_name = ''
        email = re.search(r'[\w\.-]+@[\w\.-]+', affiliation)
        email = email.group() if email else None
        parsed_authors.append({
            'first_name': first_name,
            'last_name': last_name,
            'affiliation': affiliation,
            'email': email,
            'order': index + 1
        })
    return parsed_authors

def analyze_with_openrouter(api_key, model, content, user_query):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-app-url.com",  # Replace with your app's URL
            "X-Title": "PubMed Search Analyzer",  # Replace with your app's name
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes scientific articles."},
                {"role": "user", "content": f"Analyze the following article content and determine if it's relevant to the query: '{user_query}'. If it's relevant, provide a concise summary of the main concepts, conclusions, and results. If it's not relevant, simply respond with 'Not relevant'.\n\nArticle content:\n{content}"}
            ]
        })
    )
    return response.json()['choices'][0]['message']['content']

def main():
    st.title("Enhanced PubMed Search App")

    # OpenRouter API Key input
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = ""
    st.session_state.openrouter_api_key = st.text_input("Enter your OpenRouter API Key:", type="password", value=st.session_state.openrouter_api_key)

    # Model selection
    model_options = {
        "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
        "GPT-4 Mini": "openai/gpt-4o-mini",
        "GPT-4": "openai/gpt-4o",
        "Cohere Command": "cohere/command-r",
        "Google Gemini Pro": "google/gemini-pro-1.5"
    }
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(model_options.keys())[0]
    st.session_state.selected_model = st.selectbox("Select AI Model:", list(model_options.keys()), index=list(model_options.keys()).index(st.session_state.selected_model))

    # Search parameters
    if "query" not in st.session_state:
        st.session_state.query = ""
    st.session_state.query = st.text_input("Enter your PubMed search query:", value=st.session_state.query)
    
    if "num_pages" not in st.session_state:
        st.session_state.num_pages = 1
    st.session_state.num_pages = st.number_input("Number of pages to scrape (1 page = 10 results)", min_value=1, max_value=100, value=st.session_state.num_pages)

    # Advanced search options
    with st.expander("Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.selectbox("Publication Date:", 
                                      ["Any Time", "Last Year", "Last 5 Years", "Last 10 Years", "Custom Range"])
            if date_range == "Custom Range":
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", datetime.now())
            
            article_type = st.multiselect("Article Type:", 
                                          ["Journal Article", "Clinical Trial", "Meta-Analysis", "Randomized Controlled Trial", "Review"])
        
        with col2:
            language = st.selectbox("Language:", ["Any", "English", "French", "German", "Spanish", "Chinese"])
            
            sort_by = st.selectbox("Sort Results By:", 
                                   ["Most Recent", "Best Match", "Most Cited", "Recently Added"])

    if st.button("Search PubMed") and st.session_state.query:
        # Construct filters
        filters = []
        
        if date_range != "Any Time":
            if date_range == "Last Year":
                filters.append("dates.1-year")
            elif date_range == "Last 5 Years":
                filters.append("dates.5-years")
            elif date_range == "Last 10 Years":
                filters.append("dates.10-years")
            elif date_range == "Custom Range":
                date_filter = f"custom_date_range={start_date.strftime('%Y/%m/%d')}-{end_date.strftime('%Y/%m/%d')}"
                filters.append(date_filter)
        
        if article_type:
            type_filters = [f"article_type.{t.lower().replace(' ', '-')}" for t in article_type]
            filters.extend(type_filters)
        
        if language != "Any":
            filters.append(f"language.{language.lower()}")
        
        if sort_by == "Most Recent":
            filters.append("sort=date")
        elif sort_by == "Best Match":
            filters.append("sort=relevance")
        elif sort_by == "Most Cited":
            filters.append("sort=citation")
        elif sort_by == "Recently Added":
            filters.append("sort=pubdate")

        filters_str = "&".join(filters)

        @st.cache_data
        def fetch_pubmed_results(query, filters_str, num_pages):
            return asyncio.run(scrape_pubmed(query, filters_str, num_pages))

        with st.spinner("Searching PubMed and retrieving results..."):
            st.session_state.results = fetch_pubmed_results(st.session_state.query, filters_str, st.session_state.num_pages)
            st.session_state.df = pd.DataFrame(st.session_state.results)

        if not st.session_state.df.empty:
            st.success(f"Scraped {len(st.session_state.df)} articles")
            
            st.subheader("Raw Search Results")
            display_df = st.session_state.df.copy()
            display_df['authors'] = display_df['authors'].apply(lambda x: ', '.join([author[0] for author in x]))
            st.dataframe(display_df)
            
            # Parse author information
            all_authors = []
            for _, row in st.session_state.df.iterrows():
                authors = parse_author_info(row['authors'])
                for author in authors:
                    author.update({
                        'article_url': row['url'],
                        'article_title': row['title'],
                        'background': row['background'],
                        'results': row['results'],
                        'conclusion': row['conclusion'],
                        'keywords': row['keywords'],
                        'journal': row['journal'],
                        'date': row['date'],
                        'doi': row['doi'],
                        'pmid': row['pmid'],
                        'publication_type': row['publication_type'],
                        'mesh_terms': ', '.join(row['mesh_terms']),
                        'abstract': row['abstract'],
                        'copyright': row['copyright']
                    })
                all_authors.extend(authors)
            
            st.session_state.author_df = pd.DataFrame(all_authors)
            
            st.subheader("Parsed Data with All Data Points")
            st.dataframe(st.session_state.author_df)
            
            # Display basic statistics
            st.subheader("Search Statistics")
            st.write(f"Total articles found: {len(st.session_state.df)}")
            st.write(f"Total authors: {len(st.session_state.author_df)}")
            st.write(f"Unique journals: {st.session_state.df['journal'].nunique()}")
            st.write(f"Date range: {st.session_state.df['date'].min()} to {st.session_state.df['date'].max()}")
            
            # User input for additional search term
            if "additional_search_term" not in st.session_state:
                st.session_state.additional_search_term = ""
            st.session_state.additional_search_term = st.text_input("Enter an additional search term to filter and analyze results:", value=st.session_state.additional_search_term)
            
            if st.session_state.additional_search_term and st.session_state.openrouter_api_key:
                if st.button("Analyze with AI"):
                    with st.spinner("Analyzing articles with AI..."):
                        filtered_results = []
                        for _, row in st.session_state.df.iterrows():
                            content = f"Title: {row['title']}\nAbstract: {row['abstract']}\nBackground: {row['background']}\nResults: {row['results']}\nConclusion: {row['conclusion']}"
                            analysis = analyze_with_openrouter(st.session_state.openrouter_api_key, model_options[st.session_state.selected_model], content, st.session_state.additional_search_term)
                            if analysis != "Not relevant":
                                filtered_results.append({
                                    'title': row['title'],
                                    'authors': ', '.join([author[0] for author in row['authors']]),
                                    'journal': row['journal'],
                                    'date': row['date'],
                                    'url': row['url'],
                                    'ai_analysis': analysis
                                })
                        
                        st.session_state.filtered_df = pd.DataFrame(filtered_results)
                        
                        if not st.session_state.filtered_df.empty:
                            st.subheader("Filtered and Analyzed Results")
                            st.dataframe(st.session_state.filtered_df)
                            
                            csv = st.session_state.filtered_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download filtered results as CSV",
                                data=csv,
                                file_name="pubmed_filtered_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("No articles matched the additional search term.")
            elif st.session_state.additional_search_term:
                st.warning("Please enter your OpenRouter API Key to use the AI analysis feature.")
        else:
            st.error("No results found. Please try a different query or increase the number of pages.")

if __name__ == "__main__":
    main()
