#!/usr/bin/env python3
"""
Nobel Prize Nomination Archive Explorer

A Streamlit app to query nomination counts from the Nobel Prize archive.
Supports the five original Nobel Prize categories: Physics, Chemistry, Medicine, Literature, Peace.

Note: The archive only contains data through 50 years ago (currently through 1974).
Note: Economics is not included — The Sveriges Riksbank Prize in Economic Sciences in Memory
      of Alfred Nobel was established in 1968 (first awarded 1969) by Sweden's central bank,
      not by Alfred Nobel's will, and its nomination records are not part of the public archive.
Note: The archive has known data gaps (e.g., 1973 Physics laureates are missing). These gaps
      are in the Nobel Prize Nomination Archive itself, not this application.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Optional
import json
import os
from pathlib import Path
from io import BytesIO
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

# Constants
BASE_URL = "https://www.nobelprize.org/nomination/archive/"
PRECOMPUTED_STATS_FILE = Path(__file__).parent / "precomputed_stats.json"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

CATEGORIES = {
    'all': '',
    'physics': 'phy',
    'chemistry': 'che',
    'medicine': 'med',
    'literature': 'lit',
    'peace': 'pea',
    # Note: Economics (est. 1968 by Sveriges Riksbank) is not in the archive
}

CATEGORY_NAMES = {
    'phy': 'Physics',
    'che': 'Chemistry',
    'med': 'Physiology or Medicine',
    'lit': 'Literature',
    'pea': 'Peace',
    'eco': 'Economic Sciences',
    '': 'All Categories'
}


@dataclass
class NominationEntry:
    """A single nomination entry."""
    category: str
    year: int
    other_party: str  # Who nominated them (if nominee) or who they nominated (if nominator)
    nomination_id: str


@dataclass
class NominationResult:
    """Container for nomination search results."""
    person_id: str
    name: str
    url: str
    nominee_count: int
    nominator_count: int
    nominations_as_nominee: list  # List of NominationEntry
    nominations_as_nominator: list  # List of NominationEntry
    won_prize: bool
    prize_info: Optional[dict] = None


PRIZE_CODES = {
    '': [1, 2, 3, 4, 5],  # All categories
    'phy': [1],
    'che': [2],
    'med': [3],
    'lit': [4],
    'pea': [5],
    'eco': [6]
}


def search_archive(name: str, category: str = '', year_from: str = '', year_to: str = '') -> list:
    """
    Search the Nobel Prize nomination archive using a smart sampling approach:
    1. First, sample every 5th year to quickly find people matching the name
    2. Once found, return results (full history available on person page)
    3. If not found in samples, do a full search

    Args:
        name: Name to search for (filters results client-side)
        category: Category code (phy, che, med, lit, pea, eco) or empty for all
        year_from: Start year filter (default: 1901)
        year_to: End year filter (default: 1974)

    Returns:
        List of matching person IDs and names
    """
    # Set year range
    start_year = int(year_from) if year_from else 1901
    end_year = int(year_to) if year_to else 1974

    # Get prize codes to search
    prize_codes = PRIZE_CODES.get(category, [1, 2, 3, 4, 5])

    name_lower = name.lower()
    results = {}  # Use dict to deduplicate by ID

    def search_year(year: int) -> bool:
        """Search a single year, return True if found new results."""
        found_new = False
        for prize in prize_codes:
            url = f"{BASE_URL}list.php"
            params = {'prize': prize, 'year': year}

            try:
                response = requests.get(url, params=params, headers=HEADERS, timeout=30)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if 'show_people.php?id=' in href:
                        person_id = re.search(r'id=(\d+)', href)
                        if person_id:
                            name_text = link.get_text(strip=True)
                            if name_text and name_lower in name_text.lower():
                                pid = person_id.group(1)
                                if pid not in results:
                                    results[pid] = {
                                        'id': pid,
                                        'name': name_text,
                                        'url': urljoin(BASE_URL, href)
                                    }
                                    found_new = True

                time.sleep(0.05)
            except Exception:
                pass

        return found_new

    try:
        # Search all years in range (searching is reasonably fast now)
        # We search all years to ensure we find everyone (e.g., Pierre Curie only appears in 1903)
        for year in range(start_year, end_year + 1):
            search_year(year)

        return list(results.values())

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def get_person_details(person_id: str, year_from: str = '', year_to: str = '') -> Optional[NominationResult]:
    """
    Get detailed nomination information for a person.

    Args:
        person_id: The archive ID for the person
        year_from: Optional start year filter
        year_to: Optional end year filter

    Returns:
        NominationResult with all nomination details
    """
    url = f"{BASE_URL}show_people.php?id={person_id}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()

        # Extract nomination counts
        nominee_match = re.search(r'Nominee in (\d+) nomination', page_text)
        nominator_match = re.search(r'Nominator in (\d+) nomination', page_text)

        nominee_count = int(nominee_match.group(1)) if nominee_match else 0
        nominator_count = int(nominator_match.group(1)) if nominator_match else 0

        # Extract name from page - look for h2 with the person's name
        name = ""
        for h2 in soup.find_all('h2'):
            text = h2.get_text(strip=True)
            if text and not any(x in text.lower() for x in ['share', 'archive', 'nomination']):
                name = text
                break

        # Fallback: try Firstname/Lastname pattern
        if not name or 'organisation' in name.lower():
            firstname_match = re.search(r'Firstname:\s*(\S+)', page_text)
            lastname_match = re.search(r'Lastname/org:\s*(\S+)', page_text)
            if firstname_match and lastname_match:
                name = f"{firstname_match.group(1)} {lastname_match.group(1)}"

        # Parse nomination entries
        nominations_as_nominee = []
        nominations_as_nominator = []

        # Year filter
        year_min = int(year_from) if year_from else 0
        year_max = int(year_to) if year_to else 9999

        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            if 'show.php?id=' not in href:
                continue

            nom_id_match = re.search(r'id=(\d+)', href)
            nom_id = nom_id_match.group(1) if nom_id_match else ""

            # Pattern for nominee: "Physics 1910 by Wilhelm Ostwald"
            nominee_match = re.match(r'(Physics|Chemistry|Physiology or Medicine|Medicine|Literature|Peace|Economic Sciences|Economics)\s+(\d{4})\s+by\s+(.+)', text)
            if nominee_match:
                cat = nominee_match.group(1)
                year = int(nominee_match.group(2))
                nominator = nominee_match.group(3).strip()

                if year_min <= year <= year_max:
                    nominations_as_nominee.append(NominationEntry(
                        category=cat,
                        year=year,
                        other_party=nominator,
                        nomination_id=nom_id
                    ))
                continue

            # Pattern for nominator: "Physics 1919 for Max Planck"
            nominator_match = re.match(r'(Physics|Chemistry|Physiology or Medicine|Medicine|Literature|Peace|Economic Sciences|Economics)\s+(\d{4})\s+for\s+(.+)', text)
            if nominator_match:
                cat = nominator_match.group(1)
                year = int(nominator_match.group(2))
                nominees = nominator_match.group(3).strip()

                if year_min <= year <= year_max:
                    nominations_as_nominator.append(NominationEntry(
                        category=cat,
                        year=year,
                        other_party=nominees,
                        nomination_id=nom_id
                    ))

        # Check if they won
        won_prize = "Awarded the Nobel" in page_text
        prize_info = None
        if won_prize:
            # Try standard format: "Awarded the Nobel Prize in Physics 1921"
            prize_match = re.search(r'Awarded the Nobel Prize in (\w+(?:\s+\w+)*)\s+(\d{4})', page_text)
            if prize_match:
                prize_info = {
                    'category': prize_match.group(1),
                    'year': int(prize_match.group(2))
                }
            else:
                # Try Peace Prize format: "Awarded the Nobel Peace Prize 1920"
                peace_match = re.search(r'Awarded the Nobel Peace Prize\s+(\d{4})', page_text)
                if peace_match:
                    prize_info = {
                        'category': 'Peace',
                        'year': int(peace_match.group(1))
                    }

        return NominationResult(
            person_id=person_id,
            name=name,
            url=url,
            nominee_count=nominee_count,
            nominator_count=nominator_count,
            nominations_as_nominee=nominations_as_nominee,
            nominations_as_nominator=nominations_as_nominator,
            won_prize=won_prize,
            prize_info=prize_info
        )

    except Exception as e:
        st.error(f"Error fetching details for ID {person_id}: {e}")
        return None


def search_by_year_range(category: str, year_from: int, year_to: int, 
                         min_nominations: int = 1) -> pd.DataFrame:
    """
    Search for all nominees in a category within a year range.
    Returns a DataFrame sorted by nomination count.
    """
    url = f"{BASE_URL}list.php"
    params = {
        'category': category,
        'year1': str(year_from),
        'year2': str(year_to),
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all person entries
        results = []
        seen_ids = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'show_people.php?id=' in href:
                person_id = re.search(r'id=(\d+)', href)
                if person_id and person_id.group(1) not in seen_ids:
                    seen_ids.add(person_id.group(1))
                    results.append({
                        'id': person_id.group(1),
                        'name': link.get_text(strip=True)
                    })
        
        return results
    
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def get_laureates_by_year(year: int, prize_code: int) -> list:
    """
    Get laureates (winners) for a specific year and prize category.
    Returns list of dicts with laureate info.
    """
    url = f"{BASE_URL}list.php"
    params = {'prize': prize_code, 'year': year}

    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        laureates = []
        seen_ids = set()

        # Find all person links and check if they won
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'show_people.php?id=' in href:
                person_id = re.search(r'id=(\d+)', href)
                if person_id and person_id.group(1) not in seen_ids:
                    pid = person_id.group(1)
                    seen_ids.add(pid)
                    name = link.get_text(strip=True)

                    # Check if this person won a prize by fetching their page
                    # We'll do this in bulk later to avoid too many requests here
                    laureates.append({
                        'id': pid,
                        'name': name
                    })

        return laureates

    except Exception:
        return []


def load_precomputed_stats() -> dict:
    """Load precomputed statistics from JSON file."""
    if PRECOMPUTED_STATS_FILE.exists():
        try:
            with open(PRECOMPUTED_STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_precomputed_stats(stats: dict):
    """Save precomputed statistics to JSON file."""
    try:
        with open(PRECOMPUTED_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        st.error(f"Error saving stats: {e}")


def fit_distributions(data: list) -> dict:
    """
    Fit various statistical distributions to the nomination data.
    Returns dict with fit results for each distribution.
    """
    if not data or len(data) < 3:
        return {}

    data = np.array(data)
    results = {}

    # Distributions to try
    distributions = [
        ('Normal', scipy_stats.norm),
        ('Poisson', scipy_stats.poisson),
        ('Exponential', scipy_stats.expon),
        ('Geometric', scipy_stats.geom),
        ('Log-Normal', scipy_stats.lognorm),
    ]

    for name, dist in distributions:
        try:
            if name == 'Poisson':
                # Poisson uses mean as parameter
                mu = np.mean(data)
                # Calculate log-likelihood
                ll = np.sum(scipy_stats.poisson.logpmf(data.astype(int), mu))
                results[name] = {
                    'params': {'mu (mean)': round(mu, 2)},
                    'log_likelihood': round(ll, 2),
                    'aic': round(2 * 1 - 2 * ll, 2)  # AIC = 2k - 2ln(L)
                }
            elif name == 'Geometric':
                # Geometric: p = 1/mean
                p = 1.0 / max(np.mean(data), 1)
                ll = np.sum(scipy_stats.geom.logpmf(data.astype(int), p))
                results[name] = {
                    'params': {'p': round(p, 4)},
                    'log_likelihood': round(ll, 2),
                    'aic': round(2 * 1 - 2 * ll, 2)
                }
            elif name == 'Normal':
                mu, std = scipy_stats.norm.fit(data)
                ll = np.sum(scipy_stats.norm.logpdf(data, mu, std))
                results[name] = {
                    'params': {'mean': round(mu, 2), 'std': round(std, 2)},
                    'log_likelihood': round(ll, 2),
                    'aic': round(2 * 2 - 2 * ll, 2)
                }
            elif name == 'Exponential':
                loc, scale = scipy_stats.expon.fit(data)
                ll = np.sum(scipy_stats.expon.logpdf(data, loc, scale))
                results[name] = {
                    'params': {'loc': round(loc, 2), 'scale (mean)': round(scale, 2)},
                    'log_likelihood': round(ll, 2),
                    'aic': round(2 * 2 - 2 * ll, 2)
                }
            elif name == 'Log-Normal':
                shape, loc, scale = scipy_stats.lognorm.fit(data, floc=0)
                ll = np.sum(scipy_stats.lognorm.logpdf(data, shape, loc, scale))
                results[name] = {
                    'params': {'sigma': round(shape, 2), 'scale': round(scale, 2)},
                    'log_likelihood': round(ll, 2),
                    'aic': round(2 * 2 - 2 * ll, 2)
                }
        except Exception:
            continue

    return results


# Mapping from prize category names to codes for filtering
CATEGORY_NAME_TO_CODE = {
    'Physics': 'phy',
    'Chemistry': 'che',
    'Physiology or Medicine': 'med',
    'Medicine': 'med',
    'Literature': 'lit',
    'Peace': 'pea',
    'Economic Sciences': 'eco',
    'Economics': 'eco',
}


def create_distribution_plot(data: list, category_name: str):
    """
    Create a publication-quality histogram with log-normal fit overlay.
    Matches the style from the user's Jupyter notebook.

    Args:
        data: List of nomination counts
        category_name: Name of the category for the title

    Returns:
        tuple: (matplotlib Figure object, dict of fit parameters)
    """
    x = np.array(data, dtype=float)

    # Filter out zeros, negatives, NaN, and infinite values
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) < 3:
        return None, None

    # Fit log-normal (scipy returns shape, loc, scale)
    try:
        shape, loc, scale = scipy_stats.lognorm.fit(x, floc=0)
    except Exception:
        return None, None
    dist = scipy_stats.lognorm(shape, loc=loc, scale=scale)

    # Convert scipy's parameters to μ, σ
    sigma = shape
    mu = np.log(scale)

    # Sample median
    median_sample = np.median(x)

    # Equal-tailed 68% interval (better than exp(mu±sigma) for skewed data)
    lo68, hi68 = dist.ppf([0.16, 0.84])

    # KS test
    ks_stat, ks_pval = scipy_stats.kstest(x, 'lognorm', args=(shape, loc, scale))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram: light gray (bin_width=5 to match notebook)
    bin_width = 5
    bins = np.arange(0, x.max() + bin_width, bin_width)
    ax.hist(x, bins=bins, density=True, alpha=0.4, color="gray",
            edgecolor="black", label="Data")

    # Smooth curve for log-normal fit
    xmin = np.linspace(1, x.max(), 500)
    ax.plot(xmin, dist.pdf(xmin), color="tab:blue", lw=2.5, label="Log-normal fit")

    # Median: dashed black line
    ax.axvline(median_sample, color="black", linestyle="--", lw=2,
               label=f"Median = {median_sample:.1f}")

    # Shaded 68% interval
    mask = (xmin >= lo68) & (xmin <= hi68)
    ax.fill_between(xmin[mask], 0, dist.pdf(xmin)[mask], alpha=0.25,
                    color="tab:orange", label=f"68% = [{lo68:.1f}, {hi68:.1f}]")

    # Labels and legend
    ax.set_xlabel("Number of nominations", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(f"Distribution of Total Nominations - {category_name}", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    return fig, {'mu': mu, 'sigma': sigma, 'median': median_sample,
                 'lo68': lo68, 'hi68': hi68, 'ks_stat': ks_stat, 'ks_pval': ks_pval}


def create_multi_distribution_plot(category_data: dict):
    """
    Create a plot with multiple category distributions overlaid.

    Args:
        category_data: Dict mapping category name to list of nomination counts

    Returns:
        tuple: (matplotlib Figure object, dict of fit parameters by category)
    """
    if not category_data:
        return None, None

    # Define colors for each category
    cat_colors = {
        'All Categories': '#333333',  # Dark gray/black for overall
        'Physics': '#5B76B5',
        'Chemistry': '#E6A04B',
        'Physiology or Medicine': '#6BAF6B',
        'Medicine': '#6BAF6B',
        'Literature': '#9B6B9B',
        'Peace': '#B56B6B',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    all_fit_params = {}
    max_x = 0

    # First pass: find max x value and fit distributions
    for cat_name, data in category_data.items():
        x = np.array(data, dtype=float)
        x = x[x > 0]
        if len(x) >= 3:
            max_x = max(max_x, x.max())

    if max_x == 0:
        return None, None

    # Plot each category
    for cat_name, data in category_data.items():
        x = np.array(data, dtype=float)
        x = x[x > 0]

        if len(x) < 3:
            continue

        # Fit log-normal
        shape, loc, scale = scipy_stats.lognorm.fit(x, floc=0)
        dist = scipy_stats.lognorm(shape, loc=loc, scale=scale)

        sigma = shape
        mu = np.log(scale)
        median_sample = np.median(x)
        lo68, hi68 = dist.ppf([0.16, 0.84])
        ks_stat, ks_pval = scipy_stats.kstest(x, 'lognorm', args=(shape, loc, scale))

        all_fit_params[cat_name] = {
            'mu': mu, 'sigma': sigma, 'median': median_sample,
            'lo68': lo68, 'hi68': hi68, 'ks_stat': ks_stat, 'ks_pval': ks_pval,
            'n': len(x)
        }

        color = cat_colors.get(cat_name, '#888888')

        # Plot histogram with transparency
        bin_width = 5
        bins = np.arange(0, max_x + bin_width, bin_width)
        ax.hist(x, bins=bins, density=True, alpha=0.3, color=color, edgecolor=color, linewidth=1)

        # Plot smooth fit curve
        xrange = np.linspace(1, max_x, 500)
        ax.plot(xrange, dist.pdf(xrange), color=color, lw=2.5, label=f"{cat_name} (n={len(x)}, median={median_sample:.0f})")

    ax.set_xlabel("Number of nominations", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("Distribution Comparison Across Categories", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    return fig, all_fit_params


def get_top_non_winners(category: str, year_from: int, year_to: int, top_n: int = 10, progress_callback=None) -> list:
    """
    Get the most nominated people who never won the Nobel Prize.

    Args:
        category: Category code (phy, che, med, lit, pea) or empty for all
        year_from: Start year
        year_to: End year
        top_n: Number of top non-winners to return
        progress_callback: Optional callback function for progress updates

    Returns:
        List of dicts with non-winner stats
    """
    prize_codes = PRIZE_CODES.get(category, [1, 2, 3, 4, 5])

    non_winners = {}  # person_id -> {name, nominations by category}
    total_steps = (year_to - year_from + 1) * len(prize_codes)
    current_step = 0

    for year in range(year_from, year_to + 1):
        for prize_code in prize_codes:
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

            url = f"{BASE_URL}list.php"
            params = {'prize': prize_code, 'year': year}

            try:
                response = requests.get(url, params=params, headers=HEADERS, timeout=30)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                seen_ids = set()

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if 'show_people.php?id=' in href:
                        person_id = re.search(r'id=(\d+)', href)
                        if person_id:
                            pid = person_id.group(1)
                            if pid in seen_ids:
                                continue
                            seen_ids.add(pid)

                            name = link.get_text(strip=True)

                            if pid not in non_winners:
                                non_winners[pid] = {
                                    'id': pid,
                                    'name': name,
                                    'nominations': {},  # category -> count
                                    'total': 0,
                                    'won': None  # Will check later
                                }

                time.sleep(0.02)

            except Exception:
                continue

    if progress_callback:
        progress_callback(0.5)

    # Now check each person to see if they won and count their nominations
    # Sort by estimated count first to prioritize checking high-nomination people
    person_ids = list(non_winners.keys())

    checked = 0
    confirmed_non_winners = []

    for i, pid in enumerate(person_ids):
        if progress_callback:
            progress_callback(0.5 + 0.5 * (i / len(person_ids)))

        try:
            details = get_person_details(pid)
            if details:
                if not details.won_prize:
                    # Count nominations by category
                    cat_counts = {}
                    for nom in details.nominations_as_nominee:
                        cat = nom.category
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1

                    confirmed_non_winners.append({
                        'ID': pid,
                        'Name': details.name or non_winners[pid]['name'],
                        'Total Nominations': details.nominee_count,
                        'Nominations by Category': cat_counts
                    })

            time.sleep(0.05)
            checked += 1

        except Exception:
            continue

    # Sort by total nominations and return top N
    confirmed_non_winners.sort(key=lambda x: x['Total Nominations'], reverse=True)
    return confirmed_non_winners[:top_n]


def shorten_name_for_display(name: str) -> str:
    """
    Shorten long names for histogram display.
    Removes parenthetical translations (e.g., "World Esperanto Association (Universala Esperanto Asocio)"
    becomes "World Esperanto Association").
    """
    # Remove parenthetical content (translations, alternate names)
    if '(' in name and ')' in name:
        # Extract the part before the parenthesis
        shortened = name[:name.index('(')].strip()
        if shortened:
            return shortened
    return name


def create_non_winners_plot(non_winners: list, category_name: str):
    """
    Create a bar chart of top non-winners, similar to the notebook style.

    Args:
        non_winners: List of non-winner dicts
        category_name: Name of the category

    Returns:
        matplotlib Figure object
    """
    if not non_winners:
        return None

    # Use shortened names for histogram labels
    names = [shorten_name_for_display(nw['Name']) for nw in non_winners]
    totals = [nw['Total Nominations'] for nw in non_winners]

    # Get category breakdown for stacked bars
    all_categories = set()
    for nw in non_winners:
        all_categories.update(nw.get('Nominations by Category', {}).keys())

    # Define colors for categories
    cat_colors = {
        'Physics': '#5B76B5',
        'Chemistry': '#E6A04B',
        'Physiology or Medicine': '#6BAF6B',
        'Medicine': '#6BAF6B',
        'Literature': '#9B6B9B',
        'Peace': '#B56B6B',
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create stacked bars - primary category on bottom, secondary on top
    bottom = np.zeros(len(names))

    # Determine primary category (normalize category_name to match keys)
    primary_cat = category_name
    # Map common variations
    cat_name_map = {
        'Physics': 'Physics',
        'Chemistry': 'Chemistry',
        'Medicine': 'Physiology or Medicine',
        'Physiology Or Medicine': 'Physiology or Medicine',
        'Literature': 'Literature',
        'Peace': 'Peace',
    }
    primary_cat = cat_name_map.get(category_name, category_name)

    # Sort categories: primary first (bottom), then others (top)
    sorted_cats = []
    if primary_cat in all_categories:
        sorted_cats.append(primary_cat)
    for cat in sorted(all_categories):
        if cat != primary_cat:
            sorted_cats.append(cat)

    for cat in sorted_cats:
        counts = [nw.get('Nominations by Category', {}).get(cat, 0) for nw in non_winners]
        color = cat_colors.get(cat, '#888888')
        ax.bar(names, counts, bottom=bottom, label=cat, color=color, edgecolor='black', width=0.7)
        bottom += np.array(counts)

    # Add total labels on top
    for i, total in enumerate(totals):
        ax.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel("Number of Nominations", fontsize=12)
    ax.set_title(f"Most Nominated {category_name} Nominees Who Never Won", fontsize=14)
    ax.set_ylim(0, max(totals) * 1.1)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout()
    return fig


def create_multi_non_winners_plot(category_data: dict, top_n: int = 3):
    """
    Create a comparison bar chart of top non-winners across multiple categories.

    Args:
        category_data: Dict mapping category name to list of non-winner dicts
        top_n: Number of top non-winners to show per category

    Returns:
        matplotlib Figure object
    """
    if not category_data:
        return None

    # Define colors for categories
    cat_colors = {
        'Physics': '#5B76B5',
        'Chemistry': '#E6A04B',
        'Medicine': '#6BAF6B',
        'Literature': '#9B6B9B',
        'Peace': '#B56B6B',
    }

    # Prepare data - get top N from each category
    all_entries = []
    for cat_name, non_winners in category_data.items():
        for i, nw in enumerate(non_winners[:top_n]):
            display_name = shorten_name_for_display(nw['Name'])
            # Add rank number if showing multiple per category
            if top_n > 1:
                label = f"{display_name}\n({cat_name})"
            else:
                label = f"{display_name}\n({cat_name})"
            all_entries.append({
                'name': label,
                'full_name': nw['Name'],
                'total': nw['Total Nominations'],
                'category': cat_name,
                'rank': i + 1
            })

    # Sort by total nominations descending
    all_entries.sort(key=lambda x: x['total'], reverse=True)

    names = [e['name'] for e in all_entries]
    totals = [e['total'] for e in all_entries]
    colors = [cat_colors.get(e['category'], '#888888') for e in all_entries]

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(range(len(names)), totals, color=colors, edgecolor='black', width=0.7)

    # Add total labels on top
    for i, (bar, total) in enumerate(zip(bars, totals)):
        ax.text(bar.get_x() + bar.get_width()/2, total + 1, str(total),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Total Nominations", fontsize=12)
    ax.set_title("Top Non-Winners Across Categories", fontsize=14)
    ax.set_ylim(0, max(totals) * 1.12)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors.get(cat, '#888888'), edgecolor='black', label=cat)
                       for cat in category_data.keys()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)

    plt.tight_layout()
    return fig


def add_download_buttons(fig, filename_base: str, key_suffix: str = ""):
    """
    Add download buttons for a matplotlib figure (PNG and PDF).

    Args:
        fig: matplotlib Figure object
        filename_base: Base filename without extension (e.g., "distribution_physics")
        key_suffix: Unique suffix for Streamlit widget keys to avoid duplicates
    """
    col1, col2 = st.columns(2)

    # Save to PNG buffer (high resolution)
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    png_buffer.seek(0)

    # Save to PDF buffer
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', facecolor='white')
    pdf_buffer.seek(0)

    with col1:
        st.download_button(
            label="Download PNG",
            data=png_buffer,
            file_name=f"{filename_base}.png",
            mime="image/png",
            key=f"png_{filename_base}_{key_suffix}"
        )

    with col2:
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name=f"{filename_base}.pdf",
            mime="application/pdf",
            key=f"pdf_{filename_base}_{key_suffix}"
        )


def create_nn_year_plot(nn_data: dict):
    """Create a bar plot of N.N. nominations by year."""
    if not nn_data.get('by_year'):
        return None

    years = sorted(nn_data['by_year'].keys())
    counts = [nn_data['by_year'][y] for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(y) for y in years], counts, color='#5B76B5', edgecolor='black', alpha=0.8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of N.N. Nominations", fontsize=12)
    ax.set_title("Anonymous (N.N.) Nominations by Year", fontsize=14)
    plt.xticks(rotation=45)

    # Add count labels on bars
    for i, (year, count) in enumerate(zip(years, counts)):
        ax.text(i, count + 0.3, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    return fig


def create_nn_category_plot(nn_data: dict):
    """Create a bar plot of N.N. nominations by category."""
    if not nn_data.get('by_category'):
        return None

    cat_colors = {
        'Physics': '#5B76B5',
        'Chemistry': '#E6A04B',
        'Medicine': '#6BAF6B',
        'Literature': '#9B6B9B',
        'Peace': '#B56B6B',
    }

    categories = list(nn_data['by_category'].keys())
    counts = [nn_data['by_category'][c] for c in categories]
    colors = [cat_colors.get(c, '#888888') for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, counts, color=colors, edgecolor='black')

    ax.set_ylabel("Number of N.N. Nominations", fontsize=12)
    ax.set_title("Anonymous (N.N.) Nominations by Category", fontsize=14)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    return fig


def get_nominations_to_win_stats(category: str, year_from: int, year_to: int, progress_callback=None) -> list:
    """
    Get statistics on how many nominations laureates received before winning.

    Args:
        category: Category code (phy, che, med, lit, pea) or empty for all
        year_from: Start year
        year_to: End year
        progress_callback: Optional callback function for progress updates

    Returns:
        List of dicts with laureate stats
    """
    prize_codes = PRIZE_CODES.get(category, [1, 2, 3, 4, 5])
    prize_code_to_name = {1: 'Physics', 2: 'Chemistry', 3: 'Physiology or Medicine', 4: 'Literature', 5: 'Peace', 6: 'Economic Sciences'}

    # Build list of allowed category names based on selected category
    if category:
        allowed_categories = set()
        for name, code in CATEGORY_NAME_TO_CODE.items():
            if code == category:
                allowed_categories.add(name)
    else:
        allowed_categories = None  # None means all categories allowed

    stats = []
    total_steps = (year_to - year_from + 1) * len(prize_codes)
    current_step = 0

    # Track laureates we've already processed to avoid duplicates
    processed_laureates = set()

    for year in range(year_from, year_to + 1):
        for prize_code in prize_codes:
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)

            # Get all people from this year's nominations
            url = f"{BASE_URL}list.php"
            params = {'prize': prize_code, 'year': year}

            try:
                response = requests.get(url, params=params, headers=HEADERS, timeout=30)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                seen_ids = set()

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if 'show_people.php?id=' in href:
                        person_id = re.search(r'id=(\d+)', href)
                        if person_id:
                            pid = person_id.group(1)
                            if pid in seen_ids or pid in processed_laureates:
                                continue
                            seen_ids.add(pid)

                            # Get person details to check if they won
                            details = get_person_details(pid)
                            if details and details.won_prize and details.prize_info:
                                prize_year = details.prize_info.get('year', 0)
                                prize_cat = details.prize_info.get('category', '')

                                # Filter by category if specified
                                if allowed_categories and prize_cat not in allowed_categories:
                                    continue

                                # Only include if they won within our year range
                                if year_from <= prize_year <= year_to:
                                    processed_laureates.add(pid)

                                    # Count nominations before winning
                                    nominations_before_win = [
                                        n for n in details.nominations_as_nominee
                                        if n.year <= prize_year
                                    ]

                                    stats.append({
                                        'Name': details.name,
                                        'Prize Category': prize_cat,
                                        'Year Won': prize_year,
                                        'Nominations Before Win': len(nominations_before_win),
                                        'Total Nominations': details.nominee_count,
                                        'First Nominated': min([n.year for n in details.nominations_as_nominee]) if details.nominations_as_nominee else None,
                                        'Years Nominated': prize_year - min([n.year for n in details.nominations_as_nominee]) if details.nominations_as_nominee else 0,
                                        'ID': pid
                                    })

                            time.sleep(0.1)  # Rate limiting

            except Exception:
                continue

            time.sleep(0.05)

    return stats


# Streamlit App
def main():
    st.set_page_config(
        page_title="Nobel Prize Nomination Archive Explorer",
        page_icon="🏆",
        layout="wide"
    )
    
    st.title("🏆 Nobel Prize Nomination Archive Explorer")
    
    st.markdown("""
    Query the [Nobel Prize Nomination Archive](https://www.nobelprize.org/nomination/archive/) 
    to find nomination counts for any nominee or nominator.
    
    **Note:** Due to the 50-year secrecy rule, the archive only contains data through **1974**.
    """)
    
    # Sidebar for search options
    st.sidebar.header("Search Options")

    search_type = st.sidebar.radio(
        "Search Type",
        ["By Name", "Browse by Category/Year", "Statistics"]
    )

    # Show statistics sub-menu when Statistics is selected
    if search_type == "Statistics":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Statistics Type")
        stats_type = st.sidebar.radio(
            "Select analysis",
            ["Nominations to Win", "Compare Categories", "Top Non-Winners", "Compare Non-Winners", "Anonymous Nominators (N.N.)"],
            label_visibility="collapsed"
        )
    
    if search_type == "By Name":
        st.header("Search by Name")
        
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            name = st.text_input("Enter name to search", placeholder="e.g., Einstein, Curie, Bohr")

        with col2:
            search_role = st.selectbox(
                "Search as",
                options=["Both", "Nominee", "Nominator"],
                help="Search for this person as a nominee, nominator, or both"
            )

        with col3:
            category = st.selectbox(
                "Category",
                options=list(CATEGORIES.keys()),
                format_func=lambda x: x.title() if x != 'all' else 'All Categories'
            )
        
        col3, col4 = st.columns(2)
        with col3:
            year_from = st.text_input("Year from (optional)", placeholder="e.g., 1901")
        with col4:
            year_to = st.text_input("Year to (optional)", placeholder="e.g., 1974")
        
        if st.button("Search", type="primary"):
            if not name:
                st.warning("Please enter a name to search")
            else:
                # Show search info
                yr_from = year_from if year_from else "1901"
                yr_to = year_to if year_to else "1974"
                st.info(f"Searching for '{name}' from {yr_from} to {yr_to}... This may take a moment.")

                with st.spinner(f"Searching archive for '{name}'..."):
                    results = search_archive(
                        name,
                        CATEGORIES.get(category, ''),
                        year_from,
                        year_to
                    )

                if not results:
                    st.warning(f"No people named '{name}' found in the archive")
                else:
                    st.success(f"Found {len(results)} person(s) named '{name}'")
                    
                    # Get detailed info for each result
                    detailed_results = []
                    progress = st.progress(0)
                    
                    for i, result in enumerate(results):
                        details = get_person_details(result['id'], year_from, year_to)
                        if details:
                            detailed_results.append(details)
                        progress.progress((i + 1) / len(results))
                        time.sleep(0.3)  # Rate limiting
                    
                    progress.empty()

                    # Filter results based on search role
                    if search_role == "Nominee":
                        detailed_results = [r for r in detailed_results if len(r.nominations_as_nominee) > 0]
                        sort_key = lambda x: len(x.nominations_as_nominee)
                    elif search_role == "Nominator":
                        detailed_results = [r for r in detailed_results if len(r.nominations_as_nominator) > 0]
                        sort_key = lambda x: len(x.nominations_as_nominator)
                    else:
                        sort_key = lambda x: len(x.nominations_as_nominee) + len(x.nominations_as_nominator)

                    if not detailed_results:
                        st.warning(f"No people named '{name}' found as {search_role.lower()}")

                    # Display results
                    for result in sorted(detailed_results, key=sort_key, reverse=True):
                        nominee_filtered = len(result.nominations_as_nominee)
                        nominator_filtered = len(result.nominations_as_nominator)

                        if search_role == "Nominator":
                            label = f"**{result.name or 'Unknown'}** — {nominator_filtered} nomination(s) as nominator"
                        elif search_role == "Nominee":
                            label = f"**{result.name or 'Unknown'}** — {nominee_filtered} nomination(s) as nominee"
                        else:
                            label = f"**{result.name or 'Unknown'}** — {nominee_filtered} as nominee, {nominator_filtered} as nominator"

                        with st.expander(
                            label,
                            expanded=(len(detailed_results) == 1)
                        ):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Total Nominations as Nominee", result.nominee_count)
                                st.metric("Total Nominations as Nominator", result.nominator_count)

                            with col2:
                                if result.won_prize and result.prize_info:
                                    st.success(f"Won Nobel Prize in {result.prize_info['category']} ({result.prize_info['year']})")
                                elif result.nominee_count > 0:
                                    st.info("Did not win Nobel Prize")

                            # Show nominations based on search role
                            if search_role in ["Nominee", "Both"] and result.nominations_as_nominee:
                                st.subheader(f"Nominated by ({len(result.nominations_as_nominee)} nominations)")
                                nominee_data = []
                                for nom in sorted(result.nominations_as_nominee, key=lambda x: x.year):
                                    nominee_data.append({
                                        "Year": nom.year,
                                        "Category": nom.category,
                                        "Nominated by": nom.other_party
                                    })
                                st.dataframe(pd.DataFrame(nominee_data), hide_index=True, width='stretch')

                            if search_role in ["Nominator", "Both"] and result.nominations_as_nominator:
                                st.subheader(f"Nominated others ({len(result.nominations_as_nominator)} nominations)")
                                nominator_data = []
                                for nom in sorted(result.nominations_as_nominator, key=lambda x: x.year):
                                    nominator_data.append({
                                        "Year": nom.year,
                                        "Category": nom.category,
                                        "Nominated": nom.other_party
                                    })
                                st.dataframe(pd.DataFrame(nominator_data), hide_index=True, width='stretch')

                            st.markdown(f"[View in Archive]({result.url})")
                    
                    # Summary table
                    if len(detailed_results) > 1:
                        st.subheader("Summary Table")
                        summary_data = []
                        for r in detailed_results:
                            summary_data.append({
                                "Name": r.name,
                                "As Nominee (filtered)": len(r.nominations_as_nominee),
                                "As Nominator (filtered)": len(r.nominations_as_nominator),
                                "Total Nominee": r.nominee_count,
                                "Total Nominator": r.nominator_count,
                                "Won Prize": "Yes" if r.won_prize else "No",
                            })

                        sort_col = "As Nominator (filtered)" if search_role == "Nominator" else "As Nominee (filtered)"
                        df = pd.DataFrame(summary_data).sort_values(
                            sort_col, ascending=False
                        )
                        st.dataframe(df, hide_index=True)

                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download as CSV",
                            csv,
                            f"nobel_nominations_{name.replace(' ', '_')}.csv",
                            "text/csv"
                        )
    
    elif search_type == "Browse by Category/Year":
        st.header("Browse by Category and Year")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            browse_category = st.selectbox(
                "Category",
                options=[k for k in CATEGORIES.keys() if k != 'all'],
                format_func=lambda x: x.title()
            )
        
        with col2:
            browse_year_from = st.number_input("From Year", min_value=1901, max_value=1974, value=1901)
        
        with col3:
            browse_year_to = st.number_input("To Year", min_value=1901, max_value=1974, value=1974)
        
        if st.button("Browse", type="primary"):
            with st.spinner("Fetching nominees..."):
                results = search_by_year_range(
                    CATEGORIES[browse_category],
                    browse_year_from,
                    browse_year_to
                )
            
            if not results:
                st.info("No results found")
            else:
                st.success(f"Found {len(results)} nominee(s)")
                
                # Option to get detailed counts
                if st.checkbox("Fetch detailed nomination counts (slower)"):
                    detailed_results = []
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, result in enumerate(results):
                        status.text(f"Fetching {i+1}/{len(results)}: {result['name']}")
                        details = get_person_details(result['id'])
                        if details:
                            detailed_results.append({
                                "Name": details.name or result['name'],
                                "Nominations": details.nominee_count,
                                "Won Prize": "Yes" if details.won_prize else "No",
                                "URL": details.url
                            })
                        progress.progress((i + 1) / len(results))
                        time.sleep(0.3)
                    
                    progress.empty()
                    status.empty()
                    
                    df = pd.DataFrame(detailed_results).sort_values("Nominations", ascending=False)
                    st.dataframe(df, hide_index=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download as CSV",
                        csv,
                        f"nobel_{browse_category}_{browse_year_from}-{browse_year_to}.csv",
                        "text/csv"
                    )
                else:
                    # Just show names
                    df = pd.DataFrame(results)
                    st.dataframe(df[['name']], hide_index=True)

    else:  # Statistics
        st.header("Statistics")

        # Load precomputed stats
        precomputed = load_precomputed_stats()

        if stats_type == "Nominations to Win":
            st.subheader("Nominations to Win Statistics")
            st.markdown("""
            Analyze how many nominations laureates received before winning their Nobel Prize.
            """)

            # Check if precomputed data is available
            precomputed_categories = [k for k in precomputed.keys() if not k.startswith('non_winners')]

            if precomputed_categories:
                st.success(f"Precomputed data available for: {', '.join([c.title() for c in precomputed_categories])}")
                use_precomputed = st.checkbox("Use precomputed data (instant results)", value=True)
            else:
                use_precomputed = False
                st.info("No precomputed data found. Click 'Compute & Save' to generate it for faster future access.")

            col1, col2, col3 = st.columns(3)

            with col1:
                stats_category = st.selectbox(
                    "Category",
                    options=list(CATEGORIES.keys()),
                    format_func=lambda x: x.title() if x != 'all' else 'All Categories',
                    key="stats_category"
                )

            with col2:
                stats_year_from = st.number_input(
                    "From Year",
                    min_value=1901,
                    max_value=1974,
                    value=1901,
                    key="stats_year_from"
                )

            with col3:
                stats_year_to = st.number_input(
                    "To Year",
                    min_value=1901,
                    max_value=1974,
                    value=1974,
                    key="stats_year_to"
                )

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                get_stats_btn = st.button("Get Statistics", type="primary")

            with col_btn2:
                compute_save_btn = st.button("Compute & Save (for future use)")

            def display_stats(stats, category_name):
                """Display statistics results with distribution fitting."""
                if not stats:
                    st.warning("No laureates found in this range")
                    return

                st.success(f"Found {len(stats)} laureate(s)")

                # Create DataFrame
                df = pd.DataFrame(stats)
                nom_data = df['Total Nominations'].tolist()

                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_noms = df['Total Nominations'].mean()
                    st.metric("Mean", f"{avg_noms:.2f}")
                with col2:
                    std_noms = df['Total Nominations'].std()
                    st.metric("Std Dev", f"{std_noms:.2f}")
                with col3:
                    median_noms = df['Total Nominations'].median()
                    st.metric("Median", f"{median_noms:.1f}")
                with col4:
                    single_nom = len(df[df['Total Nominations'] == 1])
                    st.metric("Single Nomination", single_nom)

                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    max_noms = df['Total Nominations'].max()
                    st.metric("Maximum", int(max_noms))
                with col6:
                    min_noms = df['Total Nominations'].min()
                    st.metric("Minimum", int(min_noms))
                with col7:
                    q75 = df['Total Nominations'].quantile(0.75)
                    st.metric("75th Percentile", f"{q75:.1f}")
                with col8:
                    q25 = df['Total Nominations'].quantile(0.25)
                    st.metric("25th Percentile", f"{q25:.1f}")

                # Log-Normal Distribution Visualization
                if len(nom_data) >= 3:
                    st.subheader("Log-Normal Distribution Fit")

                    # Create the publication-quality figure
                    fig, fit_params = create_distribution_plot(nom_data, category_name.title())

                    if fig is not None and fit_params is not None:
                        st.pyplot(fig)
                        add_download_buttons(fig, f"distribution_{stats_category}", "single")
                        plt.close(fig)

                        # Display fit parameters in a nice format
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("μ (log-mean)", f"{fit_params['mu']:.3f}")
                        with col2:
                            st.metric("σ (log-std)", f"{fit_params['sigma']:.3f}")
                        with col3:
                            ks_result = "Good fit" if fit_params['ks_pval'] > 0.05 else "Poor fit"
                            st.metric("KS p-value", f"{fit_params['ks_pval']:.3f}", delta=ks_result)

                        st.caption(f"68% of laureates received between {fit_params['lo68']:.1f} and {fit_params['hi68']:.1f} total nominations.")

                        # Check for overdispersion
                        mean_val = np.mean(nom_data)
                        var_val = np.var(nom_data)
                        if var_val > mean_val:
                            st.info(f"Data is overdispersed (variance {var_val:.1f} > mean {mean_val:.1f}), confirming log-normal is a better fit than Poisson.")
                    else:
                        st.warning("Insufficient data for distribution fitting (need at least 3 positive values).")

                # Distribution Fitting Comparison
                st.subheader("Distribution Comparison")
                fit_results = fit_distributions(nom_data)

                if fit_results:
                    # Find best fit (lowest AIC)
                    best_fit = min(fit_results.items(), key=lambda x: x[1]['aic'])
                    st.info(f"**Best fit:** {best_fit[0]} (lowest AIC = {best_fit[1]['aic']})")

                    # Show all fits in a table
                    fit_data = []
                    for name, result in sorted(fit_results.items(), key=lambda x: x[1]['aic']):
                        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
                        fit_data.append({
                            'Distribution': name,
                            'Parameters': params_str,
                            'Log-Likelihood': result['log_likelihood'],
                            'AIC': result['aic']
                        })

                    fit_df = pd.DataFrame(fit_data)
                    st.dataframe(fit_df, hide_index=True, width='stretch')

                    st.caption("AIC (Akaike Information Criterion): Lower is better. Balances fit quality with model complexity.")

                # Display laureate table
                st.subheader("Laureate Details")
                display_df = df[['Name', 'Prize Category', 'Year Won', 'Nominations Before Win', 'First Nominated', 'Years Nominated']].copy()
                display_df = display_df.sort_values('Year Won')
                st.dataframe(display_df, hide_index=True, width='stretch')

                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    csv,
                    f"nominations_to_win_{category_name}_{stats_year_from}-{stats_year_to}.csv",
                    "text/csv"
                )

            if get_stats_btn:
                if stats_year_from > stats_year_to:
                    st.error("'From Year' must be less than or equal to 'To Year'")
                else:
                    category_key = stats_category if stats_category != 'all' else 'all'

                    # Check if we can use precomputed data
                    can_use_precomputed = False
                    if use_precomputed:
                        if category_key == 'all':
                            # For "All Categories", combine all individual category data
                            individual_cats = [k for k in precomputed.keys() if not k.startswith('non_winners')]
                            if individual_cats:
                                can_use_precomputed = True
                                st.info("Using precomputed data (combining all categories)...")
                                all_stats = []
                                for cat in individual_cats:
                                    all_stats.extend(precomputed[cat])
                                # Filter by year range
                                stats = [s for s in all_stats if stats_year_from <= s['Year Won'] <= stats_year_to]
                                display_stats(stats, stats_category)
                        elif category_key in precomputed:
                            can_use_precomputed = True
                            st.info("Using precomputed data...")
                            all_stats = precomputed[category_key]
                            # Filter by year range
                            stats = [s for s in all_stats if stats_year_from <= s['Year Won'] <= stats_year_to]
                            display_stats(stats, stats_category)

                    if not can_use_precomputed:
                        # Compute fresh
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Searching for laureates... This may take a while.")

                        def update_progress(pct):
                            progress_bar.progress(pct)

                        stats = get_nominations_to_win_stats(
                            CATEGORIES.get(stats_category, ''),
                            stats_year_from,
                            stats_year_to,
                            progress_callback=update_progress
                        )

                        progress_bar.empty()
                        status_text.empty()

                        display_stats(stats, stats_category)

            if compute_save_btn:
                if stats_year_from > stats_year_to:
                    st.error("'From Year' must be less than or equal to 'To Year'")
                else:
                    category_key = stats_category if stats_category != 'all' else 'all'

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(f"Computing statistics for {stats_category.title() if stats_category != 'all' else 'All Categories'}...")

                    def update_progress(pct):
                        progress_bar.progress(pct)

                    stats = get_nominations_to_win_stats(
                        CATEGORIES.get(stats_category, ''),
                        stats_year_from,
                        stats_year_to,
                        progress_callback=update_progress
                    )

                    progress_bar.empty()
                    status_text.empty()

                    if stats:
                        # Save to precomputed data
                        precomputed[category_key] = stats
                        save_precomputed_stats(precomputed)
                        st.success(f"Saved {len(stats)} laureate records for {stats_category.title() if stats_category != 'all' else 'All Categories'}. Future queries will be instant!")

                        display_stats(stats, stats_category)
                    else:
                        st.warning("No laureates found to save")

        elif stats_type == "Compare Categories":
            st.subheader("Compare Distributions Across Categories")
            st.markdown("Overlay nomination distributions from multiple categories to compare them.")

            # Check which categories have precomputed data
            individual_cats = [k for k in precomputed.keys() if not k.startswith('non_winners')]

            if not individual_cats:
                st.warning("No precomputed data available. Please compute statistics for individual categories first.")
            else:
                # Add "All Categories" option to compare against overall distribution
                available_cats = ['all'] + individual_cats

                # Multi-select for categories
                cat_display_names = {
                    'all': 'All Categories (Overall)',
                    'physics': 'Physics',
                    'chemistry': 'Chemistry',
                    'medicine': 'Medicine',
                    'literature': 'Literature',
                    'peace': 'Peace'
                }

                selected_cats = st.multiselect(
                    "Select categories to compare",
                    options=available_cats,
                    default=['all', individual_cats[0]] if individual_cats else ['all'],
                    format_func=lambda x: cat_display_names.get(x, x.title())
                )

                if st.button("Compare Distributions", type="primary"):
                    if len(selected_cats) < 2:
                        st.warning("Please select at least 2 categories to compare.")
                    else:
                        # Gather data for each selected category
                        category_data = {}
                        for cat_key in selected_cats:
                            if cat_key == 'all':
                                # Combine all individual categories for "All"
                                all_nom_data = []
                                for ind_cat in individual_cats:
                                    cat_stats = precomputed.get(ind_cat, [])
                                    all_nom_data.extend([s['Total Nominations'] for s in cat_stats])
                                if all_nom_data:
                                    category_data['All Categories'] = all_nom_data
                            else:
                                cat_stats = precomputed.get(cat_key, [])
                                if cat_stats:
                                    nom_data = [s['Total Nominations'] for s in cat_stats]
                                    display_name = cat_display_names.get(cat_key, cat_key.title())
                                    category_data[display_name] = nom_data

                        if category_data:
                            # Create the comparison plot
                            fig, fit_params = create_multi_distribution_plot(category_data)

                            if fig is not None:
                                st.pyplot(fig)
                                add_download_buttons(fig, "distribution_comparison", "multi")
                                plt.close(fig)

                                # Show comparison table
                                st.subheader("Fit Parameters Comparison")
                                comparison_data = []
                                for cat_name, params in fit_params.items():
                                    comparison_data.append({
                                        'Category': cat_name,
                                        'N': params['n'],
                                        'Median': f"{params['median']:.1f}",
                                        'μ (log-mean)': f"{params['mu']:.3f}",
                                        'σ (log-std)': f"{params['sigma']:.3f}",
                                        '68% Interval': f"[{params['lo68']:.1f}, {params['hi68']:.1f}]",
                                        'KS p-value': f"{params['ks_pval']:.3f}"
                                    })

                                comp_df = pd.DataFrame(comparison_data)
                                st.dataframe(comp_df, hide_index=True, width='stretch')
                            else:
                                st.error("Could not generate comparison plot. Ensure categories have enough data.")
                        else:
                            st.error("No valid data found for selected categories.")

        elif stats_type == "Top Non-Winners":
            st.subheader("Top Nominated Non-Winners")
            st.markdown("Find the most nominated people who never won the Nobel Prize.")

            col_nw1, col_nw2 = st.columns(2)
            with col_nw1:
                nw_category = st.selectbox(
                    "Category",
                    options=[k for k in CATEGORIES.keys() if k != 'all'],
                    format_func=lambda x: x.title(),
                    key="nw_category"
                )
            with col_nw2:
                nw_top_n = st.number_input(
                    "Top N",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="nw_top_n"
                )

            # Check for precomputed non-winners data
            nw_key = f"non_winners_{nw_category}"
            if nw_key in precomputed:
                st.success(f"Precomputed data available for {nw_category.title()} non-winners")
                use_precomputed_nw = st.checkbox("Use precomputed non-winners data", value=True, key="use_precomputed_nw")
            else:
                use_precomputed_nw = False

            col_nw_btn1, col_nw_btn2 = st.columns(2)
            with col_nw_btn1:
                get_nw_btn = st.button("Get Top Non-Winners", type="primary", key="get_nw_btn")
            with col_nw_btn2:
                save_nw_btn = st.button("Compute & Save Non-Winners", key="save_nw_btn")

            if get_nw_btn:
                if use_precomputed_nw and nw_key in precomputed:
                    st.info("Using precomputed data...")
                    non_winners = precomputed[nw_key][:nw_top_n]
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(f"Finding top non-winners in {nw_category.title()}... This may take a while.")

                    def update_nw_progress(pct):
                        progress_bar.progress(pct)

                    non_winners = get_top_non_winners(
                        CATEGORIES.get(nw_category, ''),
                        1901,
                        1974,
                        top_n=nw_top_n,
                        progress_callback=update_nw_progress
                    )

                    progress_bar.empty()
                    status_text.empty()

                if non_winners:
                    st.success(f"Found {len(non_winners)} top non-winners")

                    # Create the plot
                    fig = create_non_winners_plot(non_winners, nw_category.title())
                    if fig:
                        st.pyplot(fig)
                        add_download_buttons(fig, f"non_winners_{nw_category}", "precomputed")
                        plt.close(fig)

                    # Show table
                    nw_df = pd.DataFrame([{
                        'Name': nw['Name'],
                        'Total Nominations': nw['Total Nominations'],
                        'Categories': ', '.join([f"{k}: {v}" for k, v in nw.get('Nominations by Category', {}).items()])
                    } for nw in non_winners])
                    st.dataframe(nw_df, hide_index=True, width='stretch')
                else:
                    st.warning("No non-winners found")

            if save_nw_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Computing top non-winners for {nw_category.title()}...")

                def update_nw_progress(pct):
                    progress_bar.progress(pct)

                non_winners = get_top_non_winners(
                    CATEGORIES.get(nw_category, ''),
                    1901,
                    1974,
                    top_n=50,  # Save more for future filtering
                    progress_callback=update_nw_progress
                )

                progress_bar.empty()
                status_text.empty()

                if non_winners:
                    precomputed[nw_key] = non_winners
                    save_precomputed_stats(precomputed)
                    st.success(f"Saved {len(non_winners)} non-winners for {nw_category.title()}")

                    # Show the plot
                    fig = create_non_winners_plot(non_winners[:nw_top_n], nw_category.title())
                    if fig:
                        st.pyplot(fig)
                        add_download_buttons(fig, f"non_winners_{nw_category}", "fresh")
                        plt.close(fig)

                    # Show table
                    nw_df = pd.DataFrame([{
                        'Name': nw['Name'],
                        'Total Nominations': nw['Total Nominations'],
                        'Categories': ', '.join([f"{k}: {v}" for k, v in nw.get('Nominations by Category', {}).items()])
                    } for nw in non_winners[:nw_top_n]])
                    st.dataframe(nw_df, hide_index=True, width='stretch')
                else:
                    st.warning("No non-winners found")

        elif stats_type == "Compare Non-Winners":
            st.subheader("Compare Non-Winners Across Categories")
            st.markdown("Compare the most nominated non-winners across different Nobel Prize categories.")

            # Category selection with multiselect
            available_nw_categories = []
            for cat in ['physics', 'chemistry', 'medicine', 'literature', 'peace']:
                nw_key = f"non_winners_{cat}"
                if nw_key in precomputed and precomputed[nw_key]:
                    available_nw_categories.append(cat)

            if not available_nw_categories:
                st.warning("No precomputed non-winners data available. Please compute non-winners for individual categories first.")
            else:
                selected_nw_cats = st.multiselect(
                    "Select categories to compare",
                    options=available_nw_categories,
                    default=available_nw_categories,
                    format_func=lambda x: x.title(),
                    key="compare_nw_categories"
                )

                top_n_per_cat = st.slider(
                    "Number of non-winners per category",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="nw_compare_top_n"
                )

                if st.button("Compare Non-Winners", type="primary", key="compare_nw_btn"):
                    if len(selected_nw_cats) < 2:
                        st.warning("Please select at least 2 categories to compare.")
                    else:
                        # Gather data from precomputed
                        category_data = {}
                        cat_name_map = {
                            'physics': 'Physics',
                            'chemistry': 'Chemistry',
                            'medicine': 'Medicine',
                            'literature': 'Literature',
                            'peace': 'Peace'
                        }

                        for cat in selected_nw_cats:
                            nw_key = f"non_winners_{cat}"
                            if nw_key in precomputed:
                                category_data[cat_name_map.get(cat, cat.title())] = precomputed[nw_key]

                        if category_data:
                            fig = create_multi_non_winners_plot(category_data, top_n=top_n_per_cat)
                            if fig:
                                st.pyplot(fig)
                                add_download_buttons(fig, "non_winners_comparison", "compare")
                                plt.close(fig)

                                # Show combined table
                                st.subheader("Non-Winners Data")
                                all_nw_data = []
                                for cat_name, non_winners in category_data.items():
                                    for i, nw in enumerate(non_winners[:top_n_per_cat]):
                                        all_nw_data.append({
                                            'Rank': i + 1,
                                            'Category': cat_name,
                                            'Name': nw['Name'],
                                            'Total Nominations': nw['Total Nominations'],
                                            'Breakdown': ', '.join([f"{k}: {v}" for k, v in nw.get('Nominations by Category', {}).items()])
                                        })

                                # Sort by total nominations
                                all_nw_data.sort(key=lambda x: x['Total Nominations'], reverse=True)
                                nw_comp_df = pd.DataFrame(all_nw_data)
                                st.dataframe(nw_comp_df, hide_index=True, width='stretch')
                            else:
                                st.error("Could not generate comparison plot.")
                        else:
                            st.error("No data available for selected categories.")

        else:  # Anonymous Nominators (N.N.)
            st.subheader("Anonymous Nominators (N.N.)")
            st.markdown("""
            Nominations where the nominator's identity is protected. **N.N.** (*Nomen Nescio* - "name unknown")
            is used in the archive to mask the names of nominators who are still living.
            """)

            # Check for precomputed N.N. data
            nn_key = "nn_data"
            if nn_key in precomputed and precomputed[nn_key].get('total', 0) > 0:
                nn_data = precomputed[nn_key]

                st.success(f"Found {nn_data['total']} anonymous (N.N.) nominations")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total N.N. Nominations", nn_data['total'])
                with col2:
                    years = list(nn_data.get('by_year', {}).keys())
                    if years:
                        st.metric("First Appeared", min(years))
                with col3:
                    st.metric("Categories", len(nn_data.get('by_category', {})))

                # Tabs for visualizations
                tab1, tab2 = st.tabs(["By Year", "By Category"])

                with tab1:
                    st.subheader("N.N. Nominations Over Time")
                    fig_year = create_nn_year_plot(nn_data)
                    if fig_year:
                        st.pyplot(fig_year)
                        add_download_buttons(fig_year, "nn_by_year", "year")
                        plt.close(fig_year)

                    # Year data table
                    if nn_data.get('by_year'):
                        year_df = pd.DataFrame([
                            {'Year': y, 'N.N. Nominations': c}
                            for y, c in sorted(nn_data['by_year'].items())
                        ])
                        st.dataframe(year_df, hide_index=True)

                with tab2:
                    st.subheader("N.N. Nominations by Category")
                    fig_cat = create_nn_category_plot(nn_data)
                    if fig_cat:
                        st.pyplot(fig_cat)
                        add_download_buttons(fig_cat, "nn_by_category", "cat")
                        plt.close(fig_cat)

                    # Category data table
                    if nn_data.get('by_category'):
                        cat_df = pd.DataFrame([
                            {'Category': c, 'N.N. Nominations': n}
                            for c, n in sorted(nn_data['by_category'].items(), key=lambda x: -x[1])
                        ])
                        st.dataframe(cat_df, hide_index=True)
            else:
                st.info("N.N. nomination data not yet precomputed. N.N. nominations are found in years 1968-1974, primarily in Physics and Chemistry.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source:** [Nobel Prize Nomination Archive](https://www.nobelprize.org/nomination/archive/)

    **Limitations:**
    - Archive data is only available through 1974 (50-year secrecy rule)
    - **Economics is not included** — The "Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel" was established in 1968 by Sweden's central bank on its 300th anniversary, not by Alfred Nobel's original 1895 bequest. Its nomination records are not part of the public archive.
    - **Known data gaps in the archive** — Some years have incomplete or missing records (e.g., the 1973 Physics laureates Josephson, Esaki, and Giaever do not appear in the archive). These gaps are in the Nobel Prize Nomination Archive itself, not this application.
    - Some historical records may be incomplete
    - Web scraping may be rate-limited
    """)


if __name__ == "__main__":
    main()
