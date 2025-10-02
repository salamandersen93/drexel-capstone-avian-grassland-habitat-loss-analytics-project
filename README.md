# drexel-capstone-avian-grassland-habitat-loss-analytics-project

<h2> Correlation, Linear Regression, and Network Analysis of Ornithological Citizen Science Data and USDA Agricultural Census Data: Grassland Bird Species</h2>

<h3> Background: </h3> Grassland bird species have sufferred a staggering 40% decline in population since 1966 due to a combination of habitat loss from climate change and conversion of native prairie grasslands to agricultural land. Understanding the population dynamics of these species and how they respond to changes in agricultural practice and methodologies could be a critical step in refining conservation efforts and a brighter future for species of critical concern.<br>

The dataset in this project was constructed as part of a Udacity Data Engineering nanodegree using python and postgreSQL. This analysis (DSCI 521 project) is a continuation of that work, and will leverage a set of techniques including correlation analysis, linear regression, and network analysis to drill into the data. The intention of this analysis is to bring to light trends which could be used to inform conservation actions by policymakers and agricultural organizations. For example, could decreasing treatment of cropland with pesticides in a given county help a given species rebound? Could incentivizing conversion of cropland to pastureland result in increased populations of a given species of critical conservation concern? 

<h3> Dataset Summary: </h3> To briefly summarize the efforts of dataset origins preceding this project: The original dataset is a merging of eBird observation data and USDA 2017 agricultural census data. eBird is a citizen science initiative owned by the Cornell Lab of Ornithology which allows birdwatchers and citizen scientists around the world to track their bird observations, including date/time, species counts, checklists comments, and other metadata. Data was acquired using an API key for 16 species which were manually selected based on subject matter knowledge and guidance from the 2019 Audubon Grassland Species Report. The date range was all months of the year 2017 and the geographical range was limited to the continental USA. USDA agricultural census data was downloaded from the USDA website and filtered to include relevant features, which are detailed in the data model below. The datasets were cleaned, wrangled, and merged to construct the data in a star schema, with an 'observation' table as the central fact table and dimension tables including checklist details, species details, FIPS details, and agricultural features.

<h3>Data Model:::</h3><br>

<b>Observation Table (Fact Table) ~3 million rows</b>
- Species Common Name
- Observation Count
- Sampling Event ID
- FIPS Code 
- Observation ID


<b>FIPS Table (Dimension)</b>
- FIPS Code
- County Name
- State Name


<b>Agricultural Table (Dimension)</b>
- FIPS_code: FIPS code (geographic information data)
- Acres of Land in Farms as Percent of Land Area in Acres: 2017
- Acres of Irrigated Land as Percent of Land in Farms Acreage: 2017
- Acres of Total Cropland as Percent of Land Area in Acres: 2017
- Acres of Harvested Cropland as Percent of Land in Farms Acreage: 2017
- Acres of All Types of Pastureland as Percent of Land in Farms Acreage: 2017
- Acres Enrolled in the Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs as Percent of Land in Farms Acreage: 2017
- Acres of Cropland and Pastureland Treated with Animal Manure as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Insects as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Nematodes as Percent of Total Cropland Acreage: 2017
- Acres of Crops Treated with Chemicals to Control Weeds, Grass, or Brush as Percent of Total Cropland Acreage: 2017
- Acres of Crops Treated with Chemicals to Control Growth, Thin Fruit, Ripen, or Defoliate as Percent of Total Cropland Acreage: 2017
- Acres Treated with Chemicals to Control Disease in Crops and Orchards as Percent of Total Cropland Acreage: 2017


<b>Species Taxonomy Data (Dimension) -- not included in this analysis </b>
- Species Common Name
- Species Scientific Name
- Species Taxonomic Order

<b>Sampling Event Table (Dimension) -- not included in this analysis </b>
- Sampling Event ID
- Event Date
- Latitude
- Longitude
- Locality
- Duration
- Observer ID
- Sampling Event Distance
- Sampling Event Duration

<b> IMPORTANT NOTE: THE DATASET EXCEEDS GITHUB LIMITS. PLEASE EMAIL DIRECTLY AT MA3775@DREXEL.EDU FOR FILES USED IN THIS ANALYSIS</B>

<h2> Exploratory Data Analysis and Project Scoping </h2>

Project scoping and exploratory data analysis can be found in scoping_and_EDA.pynb. This preliminary analysis included dataset evaluation and structural description of features, outlier detection, data quality checks, missing data checks and imputations, as well as some exploratory data analysis including higher level correlation analysis (non-numerical) and visualization of trends in the dataset. The exploratory data analysis phase let to the formulation of four core questions to drive the analysis phase of the project:

1. Are certain species observed in higher quantities in regions with more pastureland, cropland, irrigated land, or land in farms? This question could help to understand which species favor particular habitats and could help to inform locale of conservation efforts.
2. Do species observation quantities exhibit any relationships with the use of chemicals on cropland to control insects, nematodes, growth, weeds/grasses, or disease? Understanding these dynamics could bring hidden complexities to light and inform conservation decisions or incentivize certain agricultural practices.
3. Do counties with higher rates of enrollment in conservation programs have higher populations of grassland-dependent species? Understanding whether existing efforts to date have made an observable impact on species populations could inform future conservation decisions.
4. Are there certain clusters of species which tend to associate together in the same habitats or regions? Understanding this dynamic could help to drive the more granular aspects of conservation efforts and streamline efforts to include multiple species where feasible.

<h2> Data Analytics Phase </h2>

The data analysis (correlations, linear regression, and network analysis) portion can be find in the analytics_implementation.pynb file. All visualizations are included in the notebook files and not as separate images. The notebook file contains in-depth discussion of the findings, including hypothesis development and result interpretation for each of the four core questions described above.

<b> A note on limitations and shortcomings of the data: </b> There is an inherit bias to the dataset which significantly impacts correlations and model performance. The eBird data is directly reflected by the human population and number of individuals logging checklists on eBird. Thus, areas with more individuals logging eBird checklists will appear to have higher populations of species. In future works, it may be possible to weight models by the number of unique eBird observer IDs in a given county, however the data is still limited in the sense that only 16 species were included in the dataset. A complete dataset of every sampling event would be needed to gain truly accurate models of species population dynamics. Additionally, the data is limited in scope to the year 2017 and geographically to the continental USA. Expanding this to a multi-year approach could enable time series analysis of the dataset, but dataset construction would increase computational cost significantly (even with constructing the dataset on AWS using high performance clusters, compiliation of the dataset took several hours to run).


