import pandas as pd

def get_user_input():
    age = int(input("Enter your age: "))
    income = float(input("Enter your monthly income: "))
    risk_tolerance = input("Enter your risk tolerance (low, medium, high): ").strip().lower()
    num_transactions = int(input("Enter the number of transactions you want to log: "))
    
    transactions = []
    for i in range(1, num_transactions + 1):
        print(f"Transaction {i}:")
        amount = float(input("  Enter the amount spent: "))
        subcategory = input("  Enter where you spent it (subcategory): ")
        note = input("  Enter a note (if any): ")
        category = predict_category(loaded_model, loaded_custom_tokenizer, subcategory, note, amount)
        transactions.append((amount, subcategory, note, category))
    
    return age, income, risk_tolerance, transactions

def suggest_budget_and_investment(age, income, risk_tolerance, transactions, budget_percentage=0.9):
    df_transactions = pd.DataFrame(transactions, columns=['amount', 'subcategory', 'note', 'category'])
    category_totals = df_transactions.groupby('category')['amount'].sum()
    total_spending = df_transactions['amount'].sum()
    remaining_balance = income - total_spending
    
    if remaining_balance < 0:
        print("Warning: Your total spending exceeds your income!")
        remaining_balance = 0
    
    budget_suggestions = category_totals * budget_percentage
    investment_balance = remaining_balance
    
    investment_strategies = {}
    
    if investment_balance > 0:
        if risk_tolerance == 'low':
            investment_strategies = {
                'emergency_fund': investment_balance * 0.5,
                'bonds': investment_balance * 0.3,
                'savings_account': investment_balance * 0.1
            }
        elif risk_tolerance == 'medium':
            investment_strategies = {
                'emergency_fund': investment_balance * 0.3,
                'stocks': investment_balance * 0.4,
                'bonds': investment_balance * 0.1,
                'real_estate': investment_balance * 0.1
            }
        elif risk_tolerance == 'high':
            investment_strategies = {
                'stocks': investment_balance * 0.6,
                'real_estate': investment_balance * 0.2,
                'crypto': investment_balance * 0.1
            }
        
        if age < 30:
            investment_strategies['retirement_fund'] = investment_balance * 0.1
        elif 30 <= age < 50:
            investment_strategies['retirement_fund'] = investment_balance * 0.2
        else:
            investment_strategies['retirement_fund'] = investment_balance * 0.3
    
    return budget_suggestions, total_spending, remaining_balance, investment_strategies, df_transactions
