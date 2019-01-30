'''
    File name: chinook_gdotan.py
    Author: Guy Dotan
    Date: 01/27/2019
    Course: UCLA Stats 404
    Description: SQL query to rank artists with the most sales in CA.
'''

import sqlite3
import pandas as pd

# Step 1: Connect to DB
conn = sqlite3.connect("chinook.db")

# Step 2: Execute query and send results to pandas dataframe:
df = pd.read_sql_query(
                """          
                    SELECT 
                        subq_ca.invoice_year,
                        subq_ca.artist_name,
                        SUM(subq_ca.Quantity) num_sales,
                        SUM(subq_ca.UnitPrice) tot_sales
                    FROM
                        (SELECT 
                            invoice_items.InvoiceLineID,
                                invoice_items.TrackID,
                                invoice_items.UnitPrice,
                                invoice_items.Quantity,
                                invoices.Total,
                                invoices.InvoiceId AS invoice_id,
                                invoices.InvoiceDate,
                                STRFTIME( '%Y', invoices.InvoiceDate) AS invoice_year,
                                invoices.BillingCity,
                                invoices.BillingState,
                                tracks.Name AS track_name,
                                tracks.AlbumID AS album_id,
                                albums.Title AS album_title,
                                artists.ArtistID AS artist_id,
                                artists.Name AS artist_name
                        FROM
                            invoice_items
                        INNER JOIN invoices ON invoices.InvoiceId = invoice_items.InvoiceId
                        INNER JOIN tracks ON tracks.TrackId = invoice_items.TrackID
                        INNER JOIN albums ON albums.AlbumId = tracks.AlbumId
                        INNER JOIN artists ON artists.ArtistId = albums.ArtistId
                        WHERE
                            invoices.BillingState = 'CA') subq_ca
                    GROUP BY subq_ca.artist_id , subq_ca.invoice_year;                    
                 """
              ,conn)

# sort dataframe by invoice-year and tot_sales
sorted_df = df.sort_values(['invoice_year','tot_sales', 'artist_name'], ascending = [True, False, True])

# add a rank by descending sales, grouped by invoice-year
sorted_df['yr_rank'] = sorted_df.groupby("invoice_year")["tot_sales"].rank(ascending=0, method='min')

# subset the data frame by just those with a Top 3 rank.
top3_sales = sorted_df[ sorted_df['yr_rank'] <= 3 ]

print(top3_sales)