
"""
def extract_table_ocr(ocr_item):
    ## create our sorted item
    sorted_item = sorted(ocr_item, key=lambda x:(['bbox'][1], ['bbox'][0]))

    row = []
    current_row = []
    current_y  = None
    y_threshold = 10    

    for x in  sorted_item:
        y = x['bbox'][1]
        if current_y is None or  abs(y - current_y) <= y_threshold:
            current_row.append(x)
            current_y = y 
        else:
            
"""


from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date

class InvoiceLineItem(BaseModel):
    description: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    total: float = Field(..., ge=0)

    @field_validator('total')
    @classmethod
    def check_total(cls, v, info):
        """Validate that total ≈ quantity × unit_price (within rounding)."""
        values = info.data
        expected = values.get('quantity', 0) * values.get('unit_price', 0)
    
        if abs(v - expected) > 0.02:
            raise ValueError(f'Total {v} does not match qty×price = {expected}')
        return v

class Invoice(BaseModel):
    invoice_number: str = Field(..., pattern=r'^INV-\d{4}-\d+$')
    date: date
    vendor_name: str = Field(..., min_length=1)
    vendor_address: Optional[str] = None
    customer_name: str = Field(..., min_length=1)
    line_items: list[InvoiceLineItem] = Field(..., min_length=1)
    subtotal: float = Field(..., ge=0)
    tax: float = Field(..., ge=0)
    total: float = Field(..., ge=0)
    currency: str = Field(default="USD", pattern=r'^[A-Z]{3}$')

    @field_validator('total')
    @classmethod
    def check_grand_total(cls, v, info):
        """Validate that total ≈ subtotal + tax (within rounding)."""
        values = info.data
        expected = values.get('subtotal', 0) + values.get('tax', 0)
        if abs(v - expected) > 0.02:
            raise ValueError(f'Total {v} does not match subtotal+tax = {expected}')
        return v




