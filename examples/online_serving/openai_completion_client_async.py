import argparse
import asyncio
from openai import AsyncOpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    return parser.parse_args()

async def main(args):
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)

    models = await client.models.list()
    model = models.data[0].id

    async def call(prompt):
        return await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=256,
            extra_body={"confidence_threshold": 0.9},
        )

    prompt1 = "Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?\nAnswer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.\nSo, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.\nTherefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.\n#### 84\n\nQuestion: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:"
    prompt2 = "Question: Surfers enjoy going to the Rip Curl Myrtle Beach Surf Festival. There were 1500 surfers at the Festival on the first day, 600 more surfers on the second day than the first day, and 2/5 as many surfers on the third day as the first day. What is the average number of surfers at the Festival for the three days?\nAnswer: If there were 1500 surfers on the first day, the total for the second day is 1500 surfers + 600 surfers = <<1500+600=2100>>2100 surfers\nOn the third day, the number of surfers was 2/5 * 1500 surfers = <<2/5*1500=600>>600 surfers\nThe total number of surfers at the Festival in the three days was 1500 surfers + 2100 surfers + 600 surfers = <<1500+2100+600=4200>>4200 surfers\nThe average number of surfers at the Beach Surf Festival for the three days is 4200 surfers / 3 days = <<4200/3=1400>>1400 surfers/day\n#### 1400\n\nQuestion: Tim takes his 3 children trick or treating.  They are out for 4 hours.  Each hour they visited 5 houses.  Each house gives 3 treats per kid.  How many treats do his children get in total?\nAnswer: They visit 4*5=<<4*5=20>>20 houses\nThat means each child get 20*3=<<20*3=60>>60 treats\nSo in total they get 60*3=<<60*3=180>>180 treats\n#### 180\n\nQuestion: Jerry is cutting up wood for his wood-burning stove. Each pine tree makes 80 logs, each maple tree makes 60 logs, and each walnut tree makes 100 logs. If Jerry cuts up 8 pine trees, 3 maple trees, and 4 walnut trees, how many logs does he get?\nAnswer: First find the total number of pine logs by multiplying the number of trees by the number of logs per tree: 80 logs/pine * 8 pines = <<80*8=640>>640 logs\nThen do the same thing for the maple trees: 60 logs/maple * 3 maples = <<60*3=180>>180 logs\nAnd do the same thing for the walnut trees: 100 logs/walnut * 4 walnuts = <<100*4=400>>400 logs\nFinally, add up the number of logs from each type of tree to find the total number: 640 logs + 180 logs + 400 logs = <<640+180+400=1220>>1220 logs\n#### 1220\n\nQuestion: Luther designs clothes for a high fashion company. His latest line of clothing uses both silk and cashmere fabrics. There are ten pieces made with silk and half that number made with cashmere. If his latest line has thirteen pieces, how many pieces use a blend of cashmere and silk?\nAnswer: Luther has 10 / 2 = <<10/2=5>>5 pieces made with cashmere.\nHe has 13 - 10 = <<13-10=3>>3 pieces made without silk using only cashmere.\nThus, Luther has 5 - 3 = <<5-3=2>>2 pieces made using a blend of cashmere and silk.\n#### 2\n\nQuestion: At Rainbow Preschool, there are 80 students.  25% of them are half-day students and get picked up at noon, while the rest are full-day students. How many are full-day students?\nAnswer: Number of half day preschoolers is 80 x 25% = <<80*25*.01=20>>20 students.\nNumber of full day preschoolers is 80 - 20 = <<80-20=60>>60 students.\n#### 60\n\nQuestion: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\nAnswer:"

    # Schedule both completions to run concurrently
    completion1, completion2 = await asyncio.gather(
        call(prompt1),
        call(prompt2),
    )

    print("---- Completion 1 ----")
    print(completion1)
    print("---- Completion 2 ----")
    print(completion2)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
