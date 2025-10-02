# Grassland Bird Population Analytics: Agricultural Impact Assessment

> Correlation, Linear Regression, and Network Analysis of Ornithological Citizen Science Data and USDA Agricultural Census Data

---

## Overview

This project investigates the relationship between agricultural practices and grassland bird population dynamics across the continental United States. Grassland bird species have suffered a staggering 40% decline in population since 1966 due to habitat loss from climate change and conversion of native prairie grasslands to agricultural land. Understanding how these species respond to changes in agricultural practices is critical for refining conservation efforts and informing policy decisions.

The analysis employs correlation analysis, linear regression modeling, and network analysis techniques to identify actionable trends that could inform conservation strategies by policymakers and agricultural organizations.

---

## Research Questions

This analysis addresses four core questions designed to illuminate the complex relationships between agricultural practices and grassland bird populations:

### 1. Habitat Preference Analysis
**Are certain species observed in higher quantities in regions with more pastureland, cropland, irrigated land, or land in farms?**

Understanding which species favor particular habitats can help target conservation efforts to the most effective geographic locations and land types.

### 2. Chemical Treatment Impact
**Do species observation quantities exhibit any relationships with the use of chemicals on cropland to control insects, nematodes, growth, weeds/grasses, or disease?**

This analysis aims to uncover hidden complexities in how agricultural chemical usage affects bird populations, potentially informing recommendations for modified agricultural practices.

### 3. Conservation Program Efficacy
**Do counties with higher rates of enrollment in conservation programs have higher populations of grassland-dependent species?**

Evaluating whether existing conservation efforts have made observable impacts on species populations can inform the effectiveness and future direction of conservation investments.

### 4. Species Co-occurrence Patterns
**Are there certain clusters of species which tend to associate together in the same habitats or regions?**

Identifying species that co-occur could streamline conservation efforts by targeting multiple species simultaneously and revealing shared habitat requirements.

---

## Dataset Construction

### Data Sources

**eBird Citizen Science Data**
- Source: Cornell Lab of Ornithology's eBird initiative
- Temporal scope: All months of 2017
- Geographic scope: Continental United States
- Acquisition method: API access
- Species selection: 16 species manually selected based on subject matter expertise and guidance from the 2019 Audubon Grassland Species Report
- Data elements: Date/time, species counts, checklist metadata, geographic coordinates

**USDA Agricultural Census Data**
- Source: USDA 2017 Agricultural Census
- Acquisition method: Direct download from USDA website
- Geographic resolution: County level (FIPS codes)
- Focus: Agricultural practices and land use patterns relevant to grassland habitat

### Data Engineering Pipeline

The dataset was constructed as part of a Udacity Data Engineering nanodegree using Python and PostgreSQL. The engineering process involved:

1. API-based extraction of eBird observation data for target species
2. Download and filtering of USDA agricultural census data
3. Data cleaning and validation
4. Geographic alignment using FIPS codes
5. Schema design and implementation in star schema format
6. ETL pipeline execution for data integration

The resulting integrated dataset merges ornithological citizen science observations with comprehensive agricultural practice metrics at the county level, enabling analysis of relationships between farming practices and bird populations.

---

## Data Model

The dataset is structured as a star schema optimized for analytical queries:

### Observation Table (Fact Table)
**~3 million rows**

- Species Common Name
- Observation Count
- Sampling Event ID
- FIPS Code
- Observation ID (Primary Key)

### FIPS Table (Dimension)

- FIPS Code (Primary Key)
- County Name
- State Name

### Agricultural Features Table (Dimension)

- FIPS Code (Primary Key, Foreign Key to Observation)
- Acres of Land in Farms as Percent of Land Area in Acres: 2017
- Acres of Irrigated Land as Percent of Land in Farms Acreage: 2017
- Acres of Total Cropland as Percent of Land Area in Acres: 2017
- Acres of Harvested Cropland as Percent of Land in Farms Acreage: 2017
- Acres of All Types of Pastureland as Percent of Land in Farms Acreage: 2017
- Acres Enrolled in Conservation Programs (CRP, WRP, Farmable Wetlands, CREP) as Percent of Land in Farms Acreage: 2017
- Acres of Cropland and Pastureland Treated with Animal Manure as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Insects as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Nematodes as Percent of Total Cropland Acreage: 2017
- Acres of Crops Treated with Chemicals to Control Weeds, Grass, or Brush as Percent of Total Cropland Acreage: 2017
- Acres of Crops Treated with Chemicals to Control Growth, Thin Fruit, Ripen, or Defoliate as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Disease in Crops and Orchards as Percent of Total Cropland Acreage: 2017

### Species Taxonomy Table (Dimension)
*Not included in primary analysis*

- Species Common Name (Primary Key)
- Species Scientific Name
- Species Taxonomic Order

### Sampling Event Table (Dimension)
*Not included in primary analysis*

- Sampling Event ID (Primary Key)
- Event Date
- Latitude
- Longitude
- Locality
- Duration
- Observer ID
- Sampling Event Distance
- Sampling Event Duration

---

## Analytical Methodology

### Phase 1: Exploratory Data Analysis

**Location:** `scoping_and_EDA.ipynb`

The exploratory phase established data quality baselines and informed the analytical approach:

**Data Quality Assessment**
- Structural description and feature characterization
- Outlier detection and treatment strategies
- Missing data identification and imputation methods
- Data validation and integrity checks

**Preliminary Analysis**
- High-level correlation analysis across features
- Distribution visualization of key variables
- Geographic patterns in observation data
- Initial hypothesis formulation

**Outcomes:**
- Identification of the four core research questions
- Feature selection for analytical models
- Data transformation requirements
- Quality assurance protocols

### Phase 2: Statistical Analysis

**Location:** `analytics_implementation.ipynb`

The analytical phase employs three complementary methodologies:

**Correlation Analysis**
- Pearson correlation coefficients between agricultural features and species observations
- Statistical significance testing
- Identification of strong positive/negative relationships
- Multi-species comparison of correlation patterns

**Linear Regression Modeling**
- Species observation counts as dependent variables
- Agricultural features as independent variables
- Model performance evaluation (R², adjusted R², RMSE)
- Coefficient interpretation for actionable insights
- Residual analysis and assumption validation

**Network Analysis**
- Species co-occurrence network construction
- Community detection algorithms
- Centrality measures for key species identification
- Habitat association clusters

---

## Project Structure and Module Integration

### `scoping_and_EDA.ipynb`
**Purpose:** Exploratory data analysis and project scoping

**Key Functions:**
- Data loading and initial inspection
- Statistical summaries and distributions
- Missing data analysis and imputation
- Outlier detection and treatment
- Feature engineering and transformation
- Preliminary visualization of trends

**Integration Role:** Establishes data quality baselines, informs feature selection for analytical models, and generates research questions that drive subsequent analysis.

### `analytics_implementation.ipynb`
**Purpose:** Primary analytical implementation

**Key Functions:**
- Correlation matrix generation and visualization
- Linear regression model development and evaluation
- Network graph construction and analysis
- Hypothesis testing for each research question
- Result interpretation and discussion
- Visualization of findings

**Integration Role:** Executes statistical analyses on cleaned data from EDA phase, produces actionable insights, and generates visualizations for interpretation.

---

## Analytical Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                             │
│                                                                 │
│  eBird API → Species observations (2017, Continental USA)      │
│  USDA Website → Agricultural census data (County level)        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DATA ENGINEERING PIPELINE                      │
│                                                                 │
│  1. Data extraction and cleaning                               │
│  2. Geographic alignment (FIPS codes)                          │
│  3. Schema design (Star schema)                                │
│  4. ETL implementation (Python + PostgreSQL)                   │
│  5. Data validation and quality checks                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│          EXPLORATORY DATA ANALYSIS (scoping_and_EDA.ipynb)     │
│                                                                 │
│  1. Structural assessment and feature characterization         │
│  2. Outlier detection and missing data imputation             │
│  3. Preliminary correlation analysis                           │
│  4. Trend visualization                                        │
│  5. Research question formulation                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│      STATISTICAL ANALYSIS (analytics_implementation.ipynb)     │
│                                                                 │
│  Research Question 1: Habitat Preference Analysis              │
│    → Correlation analysis (species vs. land use types)        │
│    → Linear regression models                                  │
│                                                                 │
│  Research Question 2: Chemical Treatment Impact                │
│    → Correlation analysis (species vs. chemical usage)        │
│    → Regression modeling with agricultural chemicals           │
│                                                                 │
│  Research Question 3: Conservation Program Efficacy            │
│    → Comparison of observation counts in high vs. low         │
│      conservation enrollment counties                          │
│    → Statistical significance testing                          │
│                                                                 │
│  Research Question 4: Species Co-occurrence Patterns           │
│    → Network analysis of species associations                  │
│    → Community detection algorithms                            │
│    → Centrality measures and cluster identification            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                  RESULTS INTERPRETATION
                  CONSERVATION RECOMMENDATIONS
```

---

## Key Findings and Implications

All analytical results, visualizations, hypothesis testing, and detailed interpretation are contained within `analytics_implementation.ipynb`. The notebook provides:

- Statistical evidence for each research question
- Visualizations of relationships between variables
- Model performance metrics and validation
- Discussion of biological and conservation implications
- Recommendations for policymakers and conservation organizations

---

## Data Limitations and Future Work

### Known Limitations

**Sampling Bias**
The eBird dataset contains inherent bias reflecting human population distribution and observer participation. Counties with more active eBird contributors will appear to have higher bird populations, independent of actual population density. This observational bias significantly impacts correlation strength and model performance.

**Temporal Scope**
Analysis is limited to the year 2017, preventing:
- Time series analysis of population trends
- Assessment of year-over-year changes in response to agricultural practices
- Evaluation of lagged effects of conservation interventions

**Geographic Scope**
Continental USA focus excludes:
- Migratory patterns extending to Canada and Mexico
- Breeding grounds outside the study area
- Wintering habitats for migratory species

**Species Coverage**
Only 16 species were included based on conservation priority. A complete dataset including all sampling events would provide more robust population estimates but would substantially increase computational requirements.

### Future Enhancements

**Addressing Observational Bias**
- Weight models by the number of unique eBird observer IDs per county
- Incorporate eBird effort metrics (observer hours, distance traveled)
- Develop correction factors for sampling intensity

**Temporal Expansion**
- Multi-year dataset construction enabling time series analysis
- Lagged regression models to assess delayed effects of agricultural changes
- Trend analysis of population trajectories

**Enhanced Computational Infrastructure**
- Migration to cloud-based high-performance computing for larger datasets
- Optimization of ETL pipelines for expanded temporal/geographic scope
- Implementation of distributed computing frameworks

**Advanced Analytical Methods**
- Machine learning models for population prediction
- Geospatial analysis of habitat connectivity
- Bayesian hierarchical models accounting for nested geographic structures

---

## Technical Requirements

**Programming Languages:**
- Python (primary analysis)
- SQL (PostgreSQL for data engineering)

**Key Libraries:**
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib/seaborn (visualization)
- scikit-learn (statistical modeling)
- networkx (network analysis)
- scipy (statistical testing)

**Infrastructure:**
- PostgreSQL database for star schema implementation
- Jupyter Notebook environment for interactive analysis
- AWS (used for initial dataset construction)

---

## Data Access

**IMPORTANT NOTE:** The dataset exceeds GitHub file size limits and cannot be hosted in this repository.

**To obtain the data files used in this analysis, please contact:**
ma3775@drexel.edu

---

## Project Context

This analysis was completed as a capstone project for DSCI 521 (Drexel University) and represents a continuation of data engineering work completed as part of the Udacity Data Engineering nanodegree. The project demonstrates the application of statistical analysis, machine learning, and network analysis techniques to real-world conservation challenges.

---

## License

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use this analysis or dataset in your work, please cite:

```
Mike Andersen. (2025). Grassland Bird Population Analytics: Agricultural Impact Assessment.
Drexel University DSCI 521 Capstone Project.
https://github.com/[your-username]/drexel-capstone-avian-grassland-habitat-loss-analytics-project
```

---

## Contact

For questions, collaboration opportunities, or data access requests:

**Email:** mikeandersen622@gmail.com  
