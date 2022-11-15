drop table if exists processed.building_info;
drop table if exists processed.historic_sales;
drop table if exists processed.monthly_tax;
drop table if exists processed.property_transfer;
drop table if exists processed.rental_registration;

drop schema if exists processed;

create schema processed;

-- creating tables only choosing relevant prop class codes

-- Join only keeps the buildings that merge into monthlytax
create table processed.building_info as 
	(select bi.*, mt.prop_class_code, mt.class_description 
	from cleaned.building_info bi join cleaned.monthly_tax mt on bi.parcelid = mt.parcelid
	where mt.prop_class_code in ({class_codes}) 
	);

-- Join only keeps the parcels that merge into monthlytax
create table processed.historic_sales as 
	(select hs.*, mt.prop_class_code, mt.class_description 
	 from cleaned.historic_sales hs join cleaned.monthly_tax mt on hs.parcelid = mt.parcelid
	 where mt.prop_class_code in ({class_codes})
	 );
create table processed.monthly_tax as (select * from cleaned.monthly_tax);
create table processed.property_transfer as (select * from cleaned.property_transfer);
create table processed.rental_registration as (select * from cleaned.rental_registration);

-- Deal with duplicates in parcelid by assigning a building id (to be reshaped based on this ID in Python script)
alter table processed.building_info
	add column building_id int,
	add column tot_buildings int;

-- Note: ctid is a background auto-generated uid for each row
update processed.building_info bi
	set building_id = building_id2
		from (select ctid, row_number() over (partition by parcelid) as building_id2
			  from processed.building_info) bi2
	where bi.ctid = bi2.ctid;


-- creating a total_buildings field (# of distinct buildings on that parcel)
update processed.building_info bi
	set tot_buildings = tot_buildings2
		from (select parcelid, count(*) as tot_buildings2
			  from processed.building_info
			  group by parcelid) bi2
	where bi.parcelid = bi2.parcelid;

-- Filling in live_sqft = sum of all other square footage if sum > 0 and live_fsqft is zero
update processed.building_info bi
	set live_sqft = (sqft_flr1 + sqft_flr2 + attic_sqft) -- note: this logic holds for most of the data
	where bi.live_sqft = 0;

-- Removing rows that are not secondary/tertiary buildings on the parcel, and have zero on all columns (likely torn down structures)
delete from processed.building_info
	where (live_sqft + attic_sqft + sqft_flr1 + sqft_flr2 + sqft_flrh) = 0
	and tot_buildings = 1;

-- Dropping unnecessary columns in the datasets
alter table processed.building_info
	drop column sqft;