using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class BoardManager : MonoBehaviour
{
    private class Pixel
    {
        public int x, y;
        public int PixelNum;

        public Pixel(int _x, int _y, int _pixNum)
        {
            this.x = _x;
            this.y = _y;
            this.PixelNum = _pixNum;
        }
    }
    private class PixelInfo
    {
        private List<Pixel> curPixel;

        public PixelInfo()
        {
            curPixel = new List<Pixel>();
        }

        public void AddPixelInfo(int _x, int _y, int _pixNum)
        {
            curPixel.Add(new Pixel(_x, _y, _pixNum));
        }

        public List<Pixel> GetCurPixel()
        {
            return curPixel;
        }
    }

    [SerializeField]
    private int BoardSize;
    [SerializeField]
    private GameObject WaitingBoard;
    [SerializeField]
    private Sprite[] PixelColors;
    [SerializeField]
    private Camera theCamera;

    private bool IsPainting;
    private bool IsAllPainting;
    private int CurPixelColorNum;
    private int[,] CurBoardPixelNum;

    private string[,] PixelValues;
    private GameObject[,] CurBoardPixel;

    private PixelInfo TmpPixel;
    private RaycastHit HitInfo;
    private Stack<PixelInfo> PreBoardPixel;
    
    // Start is called before the first frame update
    void Start()
    {
        PreBoardPixel = new Stack<PixelInfo>();
        CurPixelColorNum = -1;
        GenerateField();
        InputPixelValues();
        
    }

    void Update()
    {
        PaintingPixel();
        ReversePixel();
    }

    // 초기화 함수들
    private void InputPixelValues()
    {
        StreamReader reader = new StreamReader(@"Datas/PixelColorValues.csv");
        PixelValues = new string[PixelColors.Length, 3];
        int _num = 0;
        while (!reader.EndOfStream)
        {
            string[] values = reader.ReadLine().Split(',');
            for (int i = 0; i < 3; i++)
                PixelValues[_num, i] = values[i];
            ++_num;
        }
    }

    private void GenerateField()
    {
        GameObject _defaultColor = Resources.Load("Prefabs/DefaultColor") as GameObject;
        CurBoardPixel = new GameObject[BoardSize, BoardSize];
        CurBoardPixelNum = new int[BoardSize, BoardSize];
        Vector3 _prePos;

        for(int i = 0; i < BoardSize; i++)
        {
            for(int j = 0; j < BoardSize; j++)
            {
                _prePos = new Vector3(j * 0.5f, i * 0.5f, 0);
                GameObject _clone = Instantiate(_defaultColor, _prePos, Quaternion.identity);
                CurBoardPixel[i, j] = _clone;
                _clone.name = i + "_" + j;
            }
        }
    }

    // 프로그램 진행 시 사용하는 함수들
    private void PaintingPixel()
    {
        if (Input.GetMouseButtonDown(0))
            TmpPixel = new PixelInfo();
        else if(Input.GetMouseButtonUp(0) && IsPainting)
        {
            if (PreBoardPixel.Count > 7)
                PreBoardPixel.Pop();

            PreBoardPixel.Push(TmpPixel);
            IsPainting = false;
        }

        if(Input.GetMouseButton(0))
        {
            Ray ray = theCamera.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray, out HitInfo, 100f))
            {
                string[] _nums = HitInfo.transform.name.Split('_');

                if(_nums[0] == "Button")
                    return;

                int _row = int.Parse(_nums[0]);
                int _col = int.Parse(_nums[1]);

                if (CurPixelColorNum == -1)
                    return;
                else if(CurBoardPixelNum[_row, _col] != CurPixelColorNum)
                {
                    if (IsAllPainting)
                    {
                        paintingFullPixel(_row, _col, CurBoardPixelNum[_row, _col], CurPixelColorNum);
                        IsAllPainting = false;
                    }
                    else
                        paintPixel(_row, _col, CurPixelColorNum, true);
                    IsPainting = true;
                }
            }
        }
    }

    private void ReversePixel()
    {
        if(Input.GetKeyDown(KeyCode.Z) && PreBoardPixel.Count > 0)
        {
            PixelInfo recentPixel = PreBoardPixel.Pop();

            foreach (Pixel _p in recentPixel.GetCurPixel())
                paintPixel(_p.x, _p.y, _p.PixelNum, false);
        }

    }

    private void paintPixel(int _x, int _y, int _pixNum, bool _isAdd)
    {
        if(_isAdd)
            TmpPixel.AddPixelInfo(_x, _y, CurBoardPixelNum[_x, _y]);
        CurBoardPixelNum[_x, _y] = _pixNum;
        CurBoardPixel[_x, _y].GetComponent<SpriteRenderer>().sprite = PixelColors[_pixNum];
    }

    private void paintingFullPixel(int _x, int _y, int _prePixNum, int _nexPixNum)
    {
        Debug.Log("Waiting...");
        Queue<Pixel> que = new Queue<Pixel>();
        int[,] _dir = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        bool[,] _visit = new bool[BoardSize, BoardSize];

        TmpPixel = new PixelInfo();
        que.Enqueue(new Pixel(_x, _y, 0));

        int nx, ny;
        while (que.Count > 0)
        {
            Pixel pre = que.Dequeue();

            if (_visit[pre.x, pre.y])
                continue;

            _visit[pre.x, pre.y] = true;
            for(int i = 0; i < 4; i ++)
            {
                nx = pre.x + _dir[i, 0];
                ny = pre.y + _dir[i, 1];
                if (nx < 0 || ny < 0 || nx >= BoardSize || ny >= BoardSize)
                    continue;
                else if (CurBoardPixelNum[nx, ny] != _prePixNum)
                    continue;

                if(!_visit[nx, ny])
                {
                    paintPixel(nx, ny, _nexPixNum, true);
                    que.Enqueue(new Pixel(nx, ny, 0));
                }
            }
        }

        PreBoardPixel.Push(TmpPixel);
        Debug.Log("End!");
    }

    public void SetPixelColor(int _num)
    {
        if (_num < PixelColors.Length)
            CurPixelColorNum = _num;
        else
            IsAllPainting = !IsAllPainting;
    }

    public void CreateImageFiles()
    {
        Debug.Log("Writing Image . . .");
        for (int i = 0; i < 3; i++)
        {
            StreamWriter textImage = new StreamWriter("Datas/CreatedImageText/Image_" + i+".csv");
            for (int j = BoardSize - 1; j >= 0; j--)
            {
                string textLine = "";
                for (int k = 0; k < BoardSize - 1; k++)
                {
                    textLine += PixelValues[CurBoardPixelNum[j, k], i] + ',';
                }
                textLine += PixelValues[CurBoardPixelNum[j, BoardSize - 1], i];
                textImage.WriteLine(textLine);
            }
            textImage.Close();
        }
        Debug.Log("Writing Image is Complete ! ! !");
        StartCoroutine(Move3DMapScene());
    }

    private IEnumerator Move3DMapScene()
    {
        string _filePath = @"Datas/CreatedImageData/prediction_type.csv";
        WaitingBoard.SetActive(true);
        while(true)
        {
            System.IO.FileInfo _imageFile = new FileInfo(_filePath);
            if (_imageFile.Exists)
                break;
            Debug.Log("Cycling...");
            yield return new WaitForSeconds(5);
        }
        SceneManager.LoadScene("MapCreatorScene");
        yield return null;
    }
}
