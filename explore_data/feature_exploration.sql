-- Explorations relating to understanding features (stemming from feature discussion on Excel sheet)

-- Attic_SQFT - what is the relationship with existence of attic?
select count(*)
from samples.labels 
where attic_sqft is null; -- there are no nulls to impute

-- Bsmt_SQFT 
select count(*)
from samples.labels 
where attic_sqft is null; -- there are no nulls to impute

-- Live_SQFT
select count(*)
from samples.labels 
where live_sqft is null;

-- Creating a prop_live_SQFT variable
with prop_dataset as (
	select live_sqft, attic_sqft, bsmt_sqft, sqft_flr1, sqft_flr2, sqft_flrh,
		  live_sqft::DECIMAL/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh) as "live_sqft_prop"
    from samples.labels
)
select min(live_sqft_prop), max(live_sqft_prop), avg(live_sqft_prop), 
	   percentile_disc(0.25) within group(order by live_sqft_prop) "perc_25",
	   percentile_disc(0.5) within group(order by live_sqft_prop) "median",
	   percentile_disc(0.75) within group(order by live_sqft_prop) "perc_75",
   	   percentile_disc(0.9) within group(order by live_sqft_prop) "perc_90"
from prop_dataset;

-- Understanding proportion > 1 cases
select live_sqft, attic_sqft, bsmt_sqft, sqft_flr1, sqft_flr2, sqft_flrh,
		  live_sqft::DECIMAL/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh) as "live_sqft_prop",
		  year_built, story_ht, appraisal_area, style, grade, basement, total_rooms, finished_sq_ft, finished_basement, garage_capacity
from samples.labels
where live_sqft::DECIMAL/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh) > 1;

-- How many of these proportion > 1 cases are there in our labeled set?
select live_sqft, attic_sqft, bsmt_sqft, sqft_flr1, sqft_flr2, sqft_flrh,
		  live_sqft::DECIMAL/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh) as "live_sqft_prop"
from samples.labels join samples.building_values bv using(parcelid)
where live_sqft::DECIMAL/(attic_sqft+bsmt_sqft+sqft_flr1+sqft_flr2+sqft_flrh) > 1;

-- Cases where other vars are missing
select count(*)
from samples.labels
where sqft_flr1 is null; -- no missing

select count(*)
from samples.labels
where sqft_flr2 is null; -- no missing

select count(*)
from samples.labels
where sqft_flrh is null; -- no missing

select count(*)
from samples.labels
where story_ht is null; -- no missing

select count(*)
from samples.labels
where year_built is null; -- no missing

select count(*)
from samples.labels
where number_of_parcels is null; -- no missing

select count(*)
from samples.labels
where appraisal_area is null; -- no missing

select count(*)
from samples.labels 
where fireplaces is null; -- no missing (same for total_rooms, full_bath, half_bath)

-- Checking style variable and how this interacts with stories
select style, story_ht
from samples.labels
where style = 'Conventional' and story_ht > 3;

-- Checking comparisons of historic sales and building info square footage variables

select live_sqft, finished_sq_ft, total_finish_area
from samples.labels;

-- Live_sqft and total_finish area are the same for all except this one observation
select count(*)
from samples.labels
where live_sqft != total_finish_area;

-- Counts of greater, less than, and equal (finished_sq_ft and live_sqft)
select count(*)
from samples.labels 
where finished_sq_ft < live_sqft;

select attic_sqft, bsmt_sqft, live_sqft, sqft_flr1, sqft_flr2, sqft_flrh, finished_sq_ft, first_floor_area, half_floor_area, finished_basement, garage_type 
from samples.labels
where finished_sq_ft > live_sqft;

select attic_sqft, bsmt_sqft, live_sqft, sqft_flr1, sqft_flr2, sqft_flrh, finished_sq_ft, first_floor_area, half_floor_area, finished_basement, garage_type 
from samples.labels
where finished_sq_ft < live_sqft;

-- Seems to be the case for most
select count(*) 
from samples.labels
where finished_sq_ft = first_floor_area and finished_sq_ft < live_sqft;

-- Same for all
select count(*)
from samples.labels
where first_floor_area != sqft_flr1;

select attic_sqft, bsmt_sqft, live_sqft, sqft_flr1, sqft_flr2, sqft_flrh, finished_sq_ft, first_floor_area, half_floor_area, finished_basement, garage_type 
from samples.labels
where half_floor_area != sqft_flrh;

-- Seems like half_floor_area is just capturing second floor area
select count(*)
from samples.labels 
where half_floor_area = sqft_flr2 + sqft_flrh;

select count(*)
from samples.labels
where finished_basement != bsmt_sqft;

select count(*)
from samples.labels 
where finished_basement > bsmt_sqft;

select bsmt_sqft, finished_basement, basement
from samples.labels 
where finished_basement != 0;

select count(*)
from samples.labels
where garage_capacity is null;





