"""
Document extraction schemas for langextract.

Each schema defines the structure to extract from different document types,
using langextract's few-shot pattern for LLM-guided extraction.
"""

from typing import Optional

# Insurance document schema
INSURANCE_SCHEMA = {
    "name": "insurance",
    "description": "Insurance policy document",
    "fields": {
        "policy_number": {
            "type": "string",
            "description": "The insurance policy number or ID"
        },
        "policy_type": {
            "type": "string",
            "description": "Type of insurance (health, auto, home, life, etc.)"
        },
        "provider": {
            "type": "string",
            "description": "Insurance company name"
        },
        "insured_name": {
            "type": "string",
            "description": "Name of the insured person"
        },
        "coverage_amount": {
            "type": "string",
            "description": "Total coverage amount or limit"
        },
        "premium": {
            "type": "string",
            "description": "Premium amount and frequency (e.g., $500/month)"
        },
        "deductible": {
            "type": "string",
            "description": "Deductible amount"
        },
        "effective_date": {
            "type": "string",
            "description": "Policy start date"
        },
        "expiration_date": {
            "type": "string",
            "description": "Policy end/renewal date"
        },
        "group_number": {
            "type": "string",
            "description": "Group number (for health insurance)"
        },
        "member_id": {
            "type": "string",
            "description": "Member ID (for health insurance)"
        }
    },
    "few_shot_examples": [
        {
            "input": "CIGNA HEALTH INSURANCE\nPolicy Number: HMO-2024-887654\nMember ID: XYZ123456789\nGroup: EMPLOYER-GROUP-001\nInsured: John Smith\nEffective: 01/01/2024\nPremium: $450.00/month\nDeductible: $1,500 individual",
            "output": {
                "policy_number": "HMO-2024-887654",
                "policy_type": "health",
                "provider": "Cigna",
                "insured_name": "John Smith",
                "premium": "$450.00/month",
                "deductible": "$1,500",
                "effective_date": "01/01/2024",
                "group_number": "EMPLOYER-GROUP-001",
                "member_id": "XYZ123456789"
            }
        }
    ]
}

# Vehicle registration schema
VEHICLE_REGISTRATION_SCHEMA = {
    "name": "vehicle_registration",
    "description": "Vehicle registration or DMV document",
    "fields": {
        "plate_number": {
            "type": "string",
            "description": "License plate number"
        },
        "registration_number": {
            "type": "string",
            "description": "Registration ID or document number"
        },
        "vin": {
            "type": "string",
            "description": "Vehicle Identification Number (17 characters)"
        },
        "vehicle_year": {
            "type": "string",
            "description": "Model year of the vehicle"
        },
        "vehicle_make": {
            "type": "string",
            "description": "Vehicle manufacturer (e.g., Toyota, Honda)"
        },
        "vehicle_model": {
            "type": "string",
            "description": "Vehicle model name"
        },
        "owner_name": {
            "type": "string",
            "description": "Registered owner name"
        },
        "registration_date": {
            "type": "string",
            "description": "Date of registration"
        },
        "expiration_date": {
            "type": "string",
            "description": "Registration expiration date"
        },
        "state": {
            "type": "string",
            "description": "State of registration"
        }
    },
    "few_shot_examples": [
        {
            "input": "NEW YORK STATE DMV\nVehicle Registration\nPlate: ABC-1234\nVIN: 1HGCM82633A123456\n2021 Tesla Model 3\nOwner: John Smith\nExpires: 03/15/2026",
            "output": {
                "plate_number": "ABC-1234",
                "vin": "1HGCM82633A123456",
                "vehicle_year": "2021",
                "vehicle_make": "Tesla",
                "vehicle_model": "Model 3",
                "owner_name": "John Smith",
                "expiration_date": "03/15/2026",
                "state": "New York"
            }
        }
    ]
}

# Receipt schema
RECEIPT_SCHEMA = {
    "name": "receipt",
    "description": "Receipt or invoice document",
    "fields": {
        "merchant_name": {
            "type": "string",
            "description": "Store or merchant name"
        },
        "merchant_address": {
            "type": "string",
            "description": "Store address"
        },
        "transaction_date": {
            "type": "string",
            "description": "Date of purchase"
        },
        "total_amount": {
            "type": "string",
            "description": "Total purchase amount"
        },
        "subtotal": {
            "type": "string",
            "description": "Subtotal before tax"
        },
        "tax_amount": {
            "type": "string",
            "description": "Tax amount"
        },
        "payment_method": {
            "type": "string",
            "description": "How payment was made (cash, credit card ending in XXXX)"
        },
        "items": {
            "type": "array",
            "description": "List of purchased items with quantities and prices"
        },
        "receipt_number": {
            "type": "string",
            "description": "Receipt or transaction number"
        }
    },
    "few_shot_examples": [
        {
            "input": "WHOLE FOODS MARKET\n123 Broadway, New York NY\nDate: 01/15/2024\nOrganic Milk $5.99\nBread $4.50\nSubtotal: $10.49\nTax: $0.93\nTotal: $11.42\nVISA ending 4567",
            "output": {
                "merchant_name": "Whole Foods Market",
                "merchant_address": "123 Broadway, New York NY",
                "transaction_date": "01/15/2024",
                "subtotal": "$10.49",
                "tax_amount": "$0.93",
                "total_amount": "$11.42",
                "payment_method": "VISA ending 4567",
                "items": ["Organic Milk $5.99", "Bread $4.50"]
            }
        }
    ]
}

# Contract schema
CONTRACT_SCHEMA = {
    "name": "contract",
    "description": "Contract or agreement document",
    "fields": {
        "contract_type": {
            "type": "string",
            "description": "Type of contract (lease, employment, service, etc.)"
        },
        "parties": {
            "type": "array",
            "description": "Names of parties involved in the contract"
        },
        "effective_date": {
            "type": "string",
            "description": "When the contract takes effect"
        },
        "termination_date": {
            "type": "string",
            "description": "When the contract ends or renews"
        },
        "payment_terms": {
            "type": "string",
            "description": "Payment amount, frequency, and due date"
        },
        "key_terms": {
            "type": "array",
            "description": "Important terms or conditions"
        },
        "renewal_terms": {
            "type": "string",
            "description": "Auto-renewal or termination notice requirements"
        }
    },
    "few_shot_examples": []
}

# Generic document schema (fallback)
GENERIC_DOCUMENT_SCHEMA = {
    "name": "generic",
    "description": "Generic document extraction",
    "fields": {
        "document_type": {
            "type": "string",
            "description": "What kind of document this is"
        },
        "title": {
            "type": "string",
            "description": "Document title or heading"
        },
        "date": {
            "type": "string",
            "description": "Any date mentioned in the document"
        },
        "key_information": {
            "type": "array",
            "description": "Most important facts or data points"
        },
        "names_mentioned": {
            "type": "array",
            "description": "People or organizations named"
        },
        "numbers": {
            "type": "array",
            "description": "Important numbers (IDs, amounts, phone numbers)"
        },
        "summary": {
            "type": "string",
            "description": "Brief summary of document contents"
        }
    },
    "few_shot_examples": []
}

# All schemas available for selection
ALL_SCHEMAS = {
    "insurance": INSURANCE_SCHEMA,
    "vehicle_registration": VEHICLE_REGISTRATION_SCHEMA,
    "receipt": RECEIPT_SCHEMA,
    "contract": CONTRACT_SCHEMA,
    "generic": GENERIC_DOCUMENT_SCHEMA
}

# Keywords that map to specific schemas
SCHEMA_KEYWORDS = {
    "insurance": [
        "insurance", "policy", "premium", "deductible", "coverage",
        "cigna", "aetna", "united", "anthem", "blue cross", "humana",
        "health plan", "dental", "vision", "medical", "member id",
        "group number"
    ],
    "vehicle_registration": [
        "registration", "dmv", "vehicle", "car", "plate", "license plate",
        "vin", "automobile", "motor vehicle", "title"
    ],
    "receipt": [
        "receipt", "purchase", "transaction", "bought", "paid", "pay",
        "store", "merchant", "invoice", "order", "cost", "total",
        "charge", "payment", "spent", "citypay", "bill"
    ],
    "contract": [
        "contract", "agreement", "lease", "rental", "employment",
        "terms", "service agreement", "subscription"
    ]
}


def get_schema_for_query(query: str) -> dict:
    """
    Select the most appropriate extraction schema based on query keywords.

    Args:
        query: Natural language question about documents

    Returns:
        Schema dict to use for extraction
    """
    query_lower = query.lower()

    # Score each schema by keyword matches
    scores = {}
    for schema_name, keywords in SCHEMA_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[schema_name] = score

    # Return highest scoring schema, or generic if no matches
    if scores:
        best_schema = max(scores, key=scores.get)
        return ALL_SCHEMAS[best_schema]

    return GENERIC_DOCUMENT_SCHEMA


def get_search_patterns_for_schema(schema_name: str) -> list[str]:
    """
    Get file search patterns relevant to a schema type.

    Args:
        schema_name: Name of the schema (e.g., 'insurance', 'vehicle_registration')

    Returns:
        List of glob patterns to search for
    """
    patterns = {
        "insurance": [
            "*insurance*", "*policy*", "*cigna*", "*aetna*", "*united*",
            "*anthem*", "*bluecross*", "*humana*", "*coverage*",
            "*health*plan*", "*dental*", "*vision*"
        ],
        "vehicle_registration": [
            "*registration*", "*dmv*", "*vehicle*", "*title*",
            "*license*", "*plate*"
        ],
        "receipt": [
            "*receipt*", "*invoice*", "*order*", "*purchase*",
            "*confirmation*"
        ],
        "contract": [
            "*contract*", "*agreement*", "*lease*", "*terms*",
            "*service*agreement*"
        ],
        "generic": ["*"]
    }

    base_patterns = patterns.get(schema_name, ["*"])

    # Add PDF extension variants
    all_patterns = []
    for p in base_patterns:
        all_patterns.append(f"{p}.pdf")
        all_patterns.append(f"{p}.PDF")
        all_patterns.append(p)

    return all_patterns


def build_extraction_prompt(schema: dict, text: str, query: Optional[str] = None) -> str:
    """
    Build a prompt for LLM extraction using the schema.

    Args:
        schema: Schema dict with fields and examples
        text: Document text to extract from
        query: Optional specific question to answer

    Returns:
        Formatted prompt for the LLM
    """
    # Truncate text if too long to avoid overwhelming the model
    if len(text) > 4000:
        text = text[:4000]

    # Build compact field list
    fields_list = ", ".join(schema["fields"].keys())

    # Simple, direct prompt that works well with thinking models
    prompt = f"""Extract these fields from the document: {fields_list}

Document:
{text}

Return ONLY JSON with the extracted values. Use null for missing fields.
JSON:"""

    return prompt
